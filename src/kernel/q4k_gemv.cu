// nova_v2/src/kernel/q4k_gemv.cu — Q4_K decode GEMV
// 一个 warp 一行；lane 固定 8 个权重 + 一组 (sc,min)；三级寄存器流水线掩盖延迟。
#include "nova/kernel.h"
#include "nova/quant/q4_k.h"

namespace nova {
namespace {

template <int K>
__global__ void quant_kernel(int8_t* __restrict__ qs, float* __restrict__ d, int* __restrict__ sums,
                             const float* __restrict__ x, int rows) {
    constexpr int NB = K / 32;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * NB) return;
    const int row = idx / NB;
    const int b   = idx - row * NB;

    const float* xp = x + (int64_t)row * K + b * 32;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xp[i]));
    const float scale = amax * (1.f / 127.f);
    const float inv   = (amax > 0.f) ? 127.f / amax : 0.f;

    int8_t* qp = qs + (int64_t)row * K + b * 32;
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) { const int v = __float2int_rn(xp[i] * inv);  qp[i] = (int8_t)v;      s0 += v; }
    #pragma unroll
    for (int i = 16; i < 32; ++i) { const int v = __float2int_rn(xp[i] * inv); qp[i] = (int8_t)v;      s1 += v; }
    d[(int64_t)row * NB + b] = scale;
    sums[(int64_t)row * NB * 2 + b * 2 + 0] = s0;
    sums[(int64_t)row * NB * 2 + b * 2 + 1] = s1;
}

template <int K>
__global__ void gemv_kernel(const block_q4_K* __restrict__ w, const int8_t* __restrict__ xq,
                            const float* __restrict__ xd, const int* __restrict__ xs,
                            float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const block_q4_K* wrow = w + (int64_t)warp * (K / kQK);

    const int sb0  = lane >> 3;
    const int c    = lane & 7;
    const int g    = c >> 1;
    const int o    = (c & 1) << 4;
    const int woff = c * 16;
    const int alo  = g * 64 + o;
    const int ahi  = alo + 32;
    const int slo  = 2 * g;

    constexpr int NSB = K / kQK;
    constexpr int NP  = (NSB + 3) / 4;

    float acc = 0.f;
    int4 w0 = *reinterpret_cast<const int4*>((wrow + sb0)->qs + woff);
    #pragma unroll 2
    for (int p = 0; p < NP; ++p) {
        const int sbi = p * 4 + sb0;
        const int nx  = (sbi + 4 < NSB) ? (sbi + 4) : (NSB - 1);
        const int4 wv = w0;
        w0 = *reinterpret_cast<const int4*>((wrow + nx)->qs + woff);
        if (sbi >= NSB) continue;

        const block_q4_K* b = wrow + sbi;
        const int4 al = *reinterpret_cast<const int4*>(xq + (int64_t)sbi * kQK + alo);
        const int4 ah = *reinterpret_cast<const int4*>(xq + (int64_t)sbi * kQK + ahi);
        const int sm_lo = xs[sbi * 16 + (alo >> 4)];
        const int sm_hi = xs[sbi * 16 + (ahi >> 4)];
        const float2 dx = *reinterpret_cast<const float2*>(xd + sbi * 8 + slo);

        uint8_t scl, mcl, sch, mch;
        get_scale_min_k4(slo,     b->scales, scl, mcl);
        get_scale_min_k4(slo + 1, b->scales, sch, mch);
        const float dd = half_to_float(b->d);
        const float dm = half_to_float(b->dmin);

        int dlo = 0, dhi = 0;
        dlo = __dp4a(wv.x & 0x0F0F0F0F, al.x, dlo);
        dlo = __dp4a(wv.y & 0x0F0F0F0F, al.y, dlo);
        dlo = __dp4a(wv.z & 0x0F0F0F0F, al.z, dlo);
        dlo = __dp4a(wv.w & 0x0F0F0F0F, al.w, dlo);
        dhi = __dp4a((wv.x >> 4) & 0x0F0F0F0F, ah.x, dhi);
        dhi = __dp4a((wv.y >> 4) & 0x0F0F0F0F, ah.y, dhi);
        dhi = __dp4a((wv.z >> 4) & 0x0F0F0F0F, ah.z, dhi);
        dhi = __dp4a((wv.w >> 4) & 0x0F0F0F0F, ah.w, dhi);

        acc += dx.x * ((dd * scl) * (float)dlo - (dm * mcl) * (float)sm_lo);
        acc += dx.y * ((dd * sch) * (float)dhi - (dm * mch) * (float)sm_hi);
    }

    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o2);
    if (lane == 0) out[warp] = acc;
}

// 4 列：权重 int4 只读一次，喂 4 个激活向量
template <int K>
__global__ void gemv4_kernel(const block_q4_K* __restrict__ w, const int8_t* __restrict__ xq,
                             const float* __restrict__ xd, const int* __restrict__ xs,
                             float* __restrict__ out, int rows) {
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const block_q4_K* wrow = w + (int64_t)warp * (K / kQK);

    const int sb0  = lane >> 3;
    const int c    = lane & 7;
    const int g_g  = c >> 1;
    const int o    = (c & 1) << 4;
    const int woff = c * 16;
    const int alo  = g_g * 64 + o;
    const int ahi  = alo + 32;
    const int slo  = 2 * g_g;

    constexpr int NSB = K / kQK;
    constexpr int NP  = (NSB + 3) / 4;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int4 w0 = *reinterpret_cast<const int4*>((wrow + sb0)->qs + woff);

    #pragma unroll 2
    for (int p = 0; p < NP; ++p) {
        const int sbi = p * 4 + sb0;
        const int nx  = (sbi + 4 < NSB) ? (sbi + 4) : (NSB - 1);
        const int4 wv = w0;
        w0 = *reinterpret_cast<const int4*>((wrow + nx)->qs + woff);
        if (sbi >= NSB) continue;

        const block_q4_K* b = wrow + sbi;
        uint8_t scl, mcl, sch, mch;
        get_scale_min_k4(slo,     b->scales, scl, mcl);
        get_scale_min_k4(slo + 1, b->scales, sch, mch);
        const float dd = half_to_float(b->d);
        const float dm = half_to_float(b->dmin);

        int qlo[4], qhi[4];
        {
            const int v0 = wv.x, v1 = wv.y, v2 = wv.z, v3 = wv.w;
            const unsigned lo[4] = {(unsigned)(v0 & 0x0F0F0F0F), (unsigned)(v1 & 0x0F0F0F0F),
                                    (unsigned)(v2 & 0x0F0F0F0F), (unsigned)(v3 & 0x0F0F0F0F)};
            const unsigned hi[4] = {(unsigned)((v0 >> 4) & 0x0F0F0F0F), (unsigned)((v1 >> 4) & 0x0F0F0F0F),
                                    (unsigned)((v2 >> 4) & 0x0F0F0F0F), (unsigned)((v3 >> 4) & 0x0F0F0F0F)};
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                qlo[t] = (int)((lo[t] & 0xF) | ((lo[t] >> 8 & 0xF) << 8) | ((lo[t] >> 16 & 0xF) << 16) | ((lo[t] >> 24 & 0xF) << 24));
                qhi[t] = (int)((hi[t] & 0xF) | ((hi[t] >> 8 & 0xF) << 8) | ((hi[t] >> 16 & 0xF) << 16) | ((hi[t] >> 24 & 0xF) << 24));
            }
        }

        #pragma unroll
        for (int c2 = 0; c2 < 4; ++c2) {
            const int8_t* xb = xq + (int64_t)c2 * K;
            const int4 al = *reinterpret_cast<const int4*>(xb + sbi * kQK + alo);
            const int4 ah = *reinterpret_cast<const int4*>(xb + sbi * kQK + ahi);
            const int sm_lo = xs[c2 * (K / 16) + sbi * 16 + (alo >> 4)];
            const int sm_hi = xs[c2 * (K / 16) + sbi * 16 + (ahi >> 4)];
            const float2 dx = *reinterpret_cast<const float2*>(xd + c2 * (K / 32) + sbi * 8 + slo);

            int dlo = 0, dhi = 0;
            dlo = __dp4a(qlo[0], al.x, dlo); dlo = __dp4a(qlo[1], al.y, dlo);
            dlo = __dp4a(qlo[2], al.z, dlo); dlo = __dp4a(qlo[3], al.w, dlo);
            dhi = __dp4a(qhi[0], ah.x, dhi); dhi = __dp4a(qhi[1], ah.y, dhi);
            dhi = __dp4a(qhi[2], ah.z, dhi); dhi = __dp4a(qhi[3], ah.w, dhi);

            acc[c2] += dx.x * ((dd * scl) * (float)dlo - (dm * mcl) * (float)sm_lo);
            acc[c2] += dx.y * ((dd * sch) * (float)dhi - (dm * mch) * (float)sm_hi);
        }
    }

    #pragma unroll
    for (int c2 = 0; c2 < 4; ++c2) {
        float v = acc[c2];
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, o2);
        if (lane == 0) out[(int64_t)c2 * rows + warp] = v;
    }
}

} // namespace

template <int K, int R>
void launch_gemv4(const Q4KWeight& w, const int8_t* xq, const float* xd, const int* xs,
                  float* out, cudaStream_t s) {
    gemv4_kernel<K><<<R / kWarpsPerBlock, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const block_q4_K*>(w.blocks), xq, xd, xs, out, R);
}

#define NOVA_INST_GEMV4(K, R) template void launch_gemv4<K, R>(const Q4KWeight&, const int8_t*, const float*, const int*, float*, cudaStream_t);
NOVA_INST_GEMV4(Model::D,         Model::NH * Model::HD)
NOVA_INST_GEMV4(Model::D,         Model::NKV * Model::HD)
NOVA_INST_GEMV4(Model::NH * Model::HD, Model::D)
NOVA_INST_GEMV4(Model::D,         Model::FFN)
NOVA_INST_GEMV4(Model::FFN,       Model::D)
NOVA_INST_GEMV4(Model::D,         Model::V)

void launch_quant(ActQ8 dst, const float* x, int rows, cudaStream_t s) {
    if (!dst.s) std::printf("[quant] NULL sums ptr! rows=%d\n", rows);
    const int total = rows * (Model::D / 32);
    quant_kernel<Model::D><<<(total + 255) / 256, 256, 0, s>>>(
        const_cast<int8_t*>(dst.qs), const_cast<float*>(dst.d), const_cast<int*>(dst.s), x, rows);
}

void launch_quant_ffn(ActQ8 dst, const float* x, int rows, cudaStream_t s) {
    const int total = rows * (Model::FFN / 32);
    quant_kernel<Model::FFN><<<(total + 255) / 256, 256, 0, s>>>(
        const_cast<int8_t*>(dst.qs), const_cast<float*>(dst.d), const_cast<int*>(dst.s), x, rows);
}

template <int K, int R>
void launch_gemv(const Q4KWeight& w, const ActQ8& x, float* out, cudaStream_t s) {
    gemv_kernel<K><<<R / kWarpsPerBlock, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const block_q4_K*>(w.blocks), x.qs, x.d, x.s, out);
}

#define NOVA_INST_GEMV(K, R) template void launch_gemv<K, R>(const Q4KWeight&, const ActQ8&, float*, cudaStream_t);

NOVA_INST_GEMV(Model::D,         Model::NH * Model::HD)
NOVA_INST_GEMV(Model::D,         Model::NKV * Model::HD)
NOVA_INST_GEMV(Model::NH * Model::HD, Model::D)
NOVA_INST_GEMV(Model::D,         Model::FFN)
NOVA_INST_GEMV(Model::FFN,       Model::D)
NOVA_INST_GEMV(Model::D,         Model::V)

} // namespace nova
