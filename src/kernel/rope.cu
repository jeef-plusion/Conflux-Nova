// nova_v2/src/kernel/rope.cu — RoPE：布局由 ArchTraits 在编译期选定
#include "nova/kernel.h"

namespace nova {
namespace {
constexpr int HD   = Model::HD;
constexpr int HALF = HD / 2;

__global__ void rope_table_kernel(float* __restrict__ cs, float* __restrict__ sn) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kCtx * HALF) return;
    const int pos = idx / HALF;
    const int i   = idx - pos * HALF;
    const float f = exp2f(-(float)i * (1.f / (float)HALF) * log2f((float)Model::theta));
    const float a = (float)pos * f;
    cs[idx] = cosf(a);
    sn[idx] = sinf(a);
}

template <bool NEOX>
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k, int64_t pos,
                            const float* __restrict__ cs, const float* __restrict__ sn) {
    const int head = blockIdx.x;
    const int i    = threadIdx.x;
    const float c = cs[pos * HALF + i];
    const float s = sn[pos * HALF + i];

    float* base = (head < Model::NH) ? q + (int64_t)head * HD
                                     : k + (int64_t)(head - Model::NH) * HD;
    const int a = NEOX ? i : (2 * i);
    const int b = NEOX ? (i + HALF) : (2 * i + 1);

    const float x0 = base[a];
    const float x1 = base[b];
    base[a] = x0 * c - x1 * s;
    base[b] = x0 * s + x1 * c;
}
template <bool NEOX>
__global__ void rope_batch_kernel(float* __restrict__ q, float* __restrict__ k, int64_t pos0,
                                  const float* __restrict__ cs, const float* __restrict__ sn) {
    const int t    = blockIdx.x;
    const int head = blockIdx.y;
    const int i    = threadIdx.x;
    const int64_t pos = pos0 + t;

    const float c = cs[pos * HALF + i];
    const float s = sn[pos * HALF + i];
    float* base = (head < Model::NH)
                ? q + ((int64_t)t * Model::NH + head) * HD
                : k + ((int64_t)t * Model::NKV + (head - Model::NH)) * HD;
    const int a = NEOX ? i : (2 * i);
    const int b = NEOX ? (i + HALF) : (2 * i + 1);

    const float x0 = base[a];
    const float x1 = base[b];
    base[a] = x0 * c - x1 * s;
    base[b] = x0 * s + x1 * c;
}

} // namespace

void launch_rope_table(float* cs, float* sn, cudaStream_t s) {
    const int total = kCtx * HALF;
    rope_table_kernel<<<(total + 255) / 256, 256, 0, s>>>(cs, sn);
}

void launch_rope_batch(float* q, float* k, int n, int64_t pos0, const float* cs, const float* sn, cudaStream_t s) {
    dim3 grid(n, Model::NH + Model::NKV);
    if constexpr (ArchTraits<typename Model::arch>::kRope == RopeLayout::Neox)
        rope_batch_kernel<true><<<grid, HALF, 0, s>>>(q, k, pos0, cs, sn);
    else
        rope_batch_kernel<false><<<grid, HALF, 0, s>>>(q, k, pos0, cs, sn);
}

void launch_rope(float* q, float* k, int64_t pos, const float* cs, const float* sn, cudaStream_t s) {
    if constexpr (ArchTraits<typename Model::arch>::kRope == RopeLayout::Neox)
        rope_kernel<true><<<Model::NH + Model::NKV, HALF, 0, s>>>(q, k, pos, cs, sn);
    else
        rope_kernel<false><<<Model::NH + Model::NKV, HALF, 0, s>>>(q, k, pos, cs, sn);
}

} // namespace nova
