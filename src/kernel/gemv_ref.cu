// nova_v2/src/kernel/q4k_gemv.cu — Q4_K decode GEMV
// 一个 warp 一行；lane 固定 8 个权重 + 一组 (sc,min)；三级寄存器流水线掩盖延迟。
#include "nova/kernel.h"
namespace nova { template <int K, int R> void launch_gemv_ref(const Q4KWeight&, const ActQ8&, float*, cudaStream_t); }
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
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int v = __float2int_rn(xp[i] * inv);
        qp[i] = (int8_t)v;
        acc += v;
    }
    d[(int64_t)row * NB + b] = scale;
    sums[(int64_t)row * NB + b] = acc;
}

template <int K>
__global__ void gemv_ref_kernel(const block_q4_K* __restrict__ w, const int8_t* __restrict__ xq,
                            const float* __restrict__ xd, const int* __restrict__ xs,
                            float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const block_q4_K* wrow = w + (int64_t)warp * (K / kQK);

    const int j    = lane >> 3;
    const int e0   = (lane & 7) * 8;
    const int base = j * 32 + (e0 & 31);
    const int hi   = e0 >> 5;
    const int xb   = j * 2 + hi;

    const int nsb = K / kQK;
    float acc = 0.f;

    uint2 q0 = *reinterpret_cast<const uint2*>(wrow->qs + base);
    uint2 q1 = *reinterpret_cast<const uint2*>((wrow + (nsb > 1 ? 1 : 0))->qs + base);

    #pragma unroll 2
    for (int sb = 0; sb < nsb; ++sb) {
        const int n2 = (sb + 2 < nsb) ? (sb + 2) : (nsb - 1);
        const block_q4_K* b = wrow + sb;
        const uint2 qn = *reinterpret_cast<const uint2*>((wrow + n2)->qs + base);
        const uint2 qv = q0;

        uint8_t sc, m;
        get_scale_min_k4(xb, b->scales, sc, m);
        const float d1 = half_to_float(b->d)    * (float)sc;
        const float m1 = half_to_float(b->dmin) * (float)m;

        uint32_t a = hi ? (qv.x >> 4) : qv.x;
        uint32_t c = hi ? (qv.y >> 4) : qv.y;
        a &= 0x0F0F0F0Fu;
        c &= 0x0F0F0F0Fu;
        const int qlo = (int)((a & 0xF) | ((a >> 8 & 0xF) << 8) | ((a >> 16 & 0xF) << 16) | ((a >> 24 & 0xF) << 24));
        const int qhi = (int)((c & 0xF) | ((c >> 8 & 0xF) << 8) | ((c >> 16 & 0xF) << 16) | ((c >> 24 & 0xF) << 24));

        const int8_t* xp = xq + (int64_t)sb * kQK + j * 64 + e0;
        const uint2 xv = *reinterpret_cast<const uint2*>(xp);
        const float xs_v = xd[sb * 8 + xb];
        const int dot = __dp4a(qlo, (int)xv.x, 0) + __dp4a(qhi, (int)xv.y, 0);
        const int sx  = __dp4a((int)xv.x, 0x01010101, 0) + __dp4a((int)xv.y, 0x01010101, 0);
        acc = fmaf(xs_v, fmaf(d1, (float)dot, -m1 * (float)sx), acc);

        q0 = q1;
        q1 = qn;
    }

    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o2);
    if (lane == 0) out[warp] = acc;
}

} // namespace



template <int K, int R>
void launch_gemv_ref(const Q4KWeight& w, const ActQ8& x, float* out, cudaStream_t s) {
    gemv_ref_kernel<K><<<R / kWarpsPerBlock, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const block_q4_K*>(w.blocks), x.qs, x.d, x.s, out);
}

#define NOVA_INST_GEMV(K, R) template void launch_gemv_ref<K, R>(const Q4KWeight&, const ActQ8&, float*, cudaStream_t);

NOVA_INST_GEMV(Model::D,         Model::NH * Model::HD)
NOVA_INST_GEMV(Model::D,         Model::NKV * Model::HD)
NOVA_INST_GEMV(Model::NH * Model::HD, Model::D)
NOVA_INST_GEMV(Model::D,         Model::FFN)
NOVA_INST_GEMV(Model::FFN,       Model::D)
NOVA_INST_GEMV(Model::D,         Model::V)

} // namespace nova
