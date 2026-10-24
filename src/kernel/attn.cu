// nova_v2/src/kernel/attn.cu — decode: KV 切段并行 + combine；prefill: flash
// split-K 结构参考 Conflux-Nova 的 fa_split/fa_combine：part[head][split][m,l,acc...]
#include "nova/kernel.h"

namespace nova {
namespace {
constexpr int NH  = Model::NH;
constexpr int NKV = Model::NKV;
constexpr int HD  = Model::HD;
constexpr int ND  = HD / 32;
constexpr int MAXSPLIT = 32;

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);
}

// ---------------- prefill / 短序列 ----------------
template <bool SWA>
__global__ void flash_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                             const float* __restrict__ vc, float* __restrict__ out,
                             int n_q, int n_past, float scale) {
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp >= NH * n_q) return;

    const int head = warp / n_q;
    const int qi   = warp - head * n_q;
    const int kvh  = head / (NH / NKV);
    const int64_t pos = (int64_t)n_past + qi;

    const float* qp = q + ((int64_t)qi * NH + head) * HD;
    float qreg[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) qreg[t] = (t < ND) ? qp[lane + 32 * t] : 0.f;

    float m = -INFINITY, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    int64_t start = 0;
    if constexpr (SWA) start = (pos - kCtx + 1 > 0) ? pos - kCtx + 1 : 0;

    for (int64_t p = start; p <= pos; ++p) {
        const float* kp = kc + (p * NKV + kvh) * HD;
        float part = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) if (t < ND) part = fmaf(qreg[t], kp[lane + 32 * t], part);
        const float qk = warp_sum(part) * scale;

        const float mn = fmaxf(m, qk);
        const float e  = __expf(m - mn);
        const float pe = __expf(qk - mn);
        l = fmaf(l, e, pe);

        const float* vp = vc + (p * NKV + kvh) * HD;
        #pragma unroll
        for (int t = 0; t < 4; ++t) if (t < ND) acc[t] = fmaf(acc[t], e, pe * vp[lane + 32 * t]);
        m = mn;
    }

    float* op = out + ((int64_t)qi * NH + head) * HD;
    const float inv = 1.f / l;
    #pragma unroll
    for (int t = 0; t < 4; ++t) if (t < ND) op[lane + 32 * t] = acc[t] * inv;
}

// ---------------- decode: split-K ----------------
// grid (nsplit, NH)；每 block 4 个 warp，warp0 算这一段，其余 warp 负责把 K/V 搬进 smem
constexpr int BKV = 32;
constexpr int SD  = HD + 1;

__global__ void fa_split_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                const float* __restrict__ vc, float* __restrict__ part,
                                int n_keys, int chunk, float scale, int nsplit) {
    __shared__ float s_q[HD];
    __shared__ float spart[4][HD + 2];

    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int head = (int)blockIdx.y;
    const int sp   = (int)blockIdx.x;
    const int kvh  = head / (NH / NKV);

    const int q4 = (chunk + 3) >> 2;                 // 每个 warp 扫 1/4 段
    const int kv0 = sp * chunk + warp * q4;
    int kv1 = kv0 + q4;
    if (kv1 > n_keys) kv1 = n_keys;
    if (kv1 > (sp + 1) * chunk) kv1 = (sp + 1) * chunk;

    if (tid < HD) s_q[tid] = q[head * HD + tid];
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    const float* qh = s_q;
    for (int p = kv0; p < kv1; ++p) {
        const float* kp = kc + ((int64_t)p * NKV + kvh) * HD;
        float part_d = 0.f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) part_d = fmaf(qh[lane + 32 * c], kp[lane + 32 * c], part_d);
        const float qk = warp_sum(part_d) * scale;

        const float mn = fmaxf(m, qk);
        const float corr = __expf(m - mn);
        const float pr = __expf(qk - mn);
        l = fmaf(l, corr, pr);

        const float* vp = vc + ((int64_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[c] = fmaf(acc[c], corr, pr * vp[lane + 32 * c]);
        m = mn;
    }

    // block 内 4 个 warp 先合并，避免把 4 份 partial 全写回显存
    if (lane == 0) { spart[warp][0] = m; spart[warp][1] = l; }
    #pragma unroll
    for (int c = 0; c < 4; ++c) spart[warp][2 + lane + c * 32] = acc[c];
    __syncthreads();

    if (warp == 0) {
        float mm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < 4; ++w) mm = fmaxf(mm, spart[w][0]);
        float ll = 0.f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float ls = spart[w][1];
            if (ls > 0.f) ll = fmaf(ls, __expf(spart[w][0] - mm), ll);
        }
        float* out = part + ((size_t)head * nsplit + sp) * (HD + 2);
        if (lane == 0) { out[0] = mm; out[1] = ll; }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int dc = 2 + lane + c * 32;
            float a = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) a = fmaf(spart[w][dc], __expf(spart[w][0] - mm), a);
            out[dc] = a;
        }
    }
}

__global__ void fa_combine_kernel(const float* __restrict__ part, float* __restrict__ O, int nsplit) {
    const int head = (int)blockIdx.x;
    const int tid  = (int)threadIdx.x;
    const float* base = part + (size_t)head * nsplit * (HD + 2);
    __shared__ float s_m, s_l;

    if (tid == 0) {
        float m = -INFINITY;
        for (int s = 0; s < nsplit; ++s) m = fmaxf(m, base[(size_t)s * (HD + 2) + 0]);
        float l = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float ms = base[(size_t)s * (HD + 2) + 0];
            const float ls = base[(size_t)s * (HD + 2) + 1];
            if (ls > 0.f) l = fmaf(ls, __expf(ms - m), l);
        }
        s_m = m;
        s_l = l;
    }
    __syncthreads();
    for (int d = tid; d < HD; d += (int)blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float ms = base[(size_t)s * (HD + 2) + 0];
            const float av = base[(size_t)s * (HD + 2) + 2 + d];
            a = fmaf(av, __expf(ms - s_m), a);
        }
        O[head * HD + d] = (s_l > 0.f) ? a / s_l : 0.f;
    }
}

} // namespace

void launch_attn_prefill(const float* q, const float* kc, const float* vc, float* o,
                         int n_q, int n_past, cudaStream_t s) {
    constexpr float scale = (HD == 64) ? 0.125f : 0.08838834764831845f;
    const int warps = NH * n_q;
    if constexpr (ArchTraits<typename Model::arch>::kMask == MaskKind::Causal)
        flash_kernel<false><<<(warps + 3) / 4, 128, 0, s>>>(q, kc, vc, o, n_q, n_past, scale);
    else
        flash_kernel<true><<<(warps + 3) / 4, 128, 0, s>>>(q, kc, vc, o, n_q, n_past, scale);
}

void launch_attn_decode(const float* q, const float* kc, const float* vc, float* o,
                        int n_past, float* scratch, cudaStream_t s) {
    constexpr float scale = (HD == 64) ? 0.125f : 0.08838834764831845f;
    const int n_keys = n_past + 1;

    int nsplit = (n_keys + 31) / 32;
    if (nsplit > MAXSPLIT) nsplit = MAXSPLIT;
    if (nsplit < 1) nsplit = 1;
    const int chunk = (n_keys + nsplit - 1) / nsplit;

    dim3 grid((unsigned)nsplit, (unsigned)NH);
    fa_split_kernel<<<grid, 128, 0, s>>>(q, kc, vc, scratch, n_keys, chunk, scale, nsplit);
    fa_combine_kernel<<<NH, HD, 0, s>>>(scratch, o, nsplit);
}

} // namespace nova
