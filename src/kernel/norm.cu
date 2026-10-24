// nova_v2/src/kernel/norm.cu — RMSNorm / 残差 / SwiGLU+量化 / greedy
#include "nova/kernel.h"

namespace nova {
namespace {

constexpr int D = Model::D;

__global__ void rms_norm_kernel(float* __restrict__ y, const float* __restrict__ x,
                                const float* __restrict__ w) {
    const int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;

    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);

    __shared__ float sh[kWarpsPerBlock];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < kWarpsPerBlock) ? sh[threadIdx.x] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) sh[0] = t;
    }
    __syncthreads();

    const float scale = rsqrtf(sh[0] * (1.f / (float)D) + Model::eps);
    float* yr = y + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

__global__ void add_kernel(float* __restrict__ acc, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += x[i];
}

__global__ void swiglu_kernel(float* __restrict__ y, const float* __restrict__ g,
                              const float* __restrict__ u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    y[i] = (v / (1.f + __expf(-v))) * u[i];
}

__global__ void greedy_kernel(const float* __restrict__ logits, int* __restrict__ out) {
    constexpr int V = Model::V;
    float best = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; bi = i; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_down_sync(0xffffffffu, best, o);
        const int   j = __shfl_down_sync(0xffffffffu, bi, o);
        if (v > best) { best = v; bi = j; }
    }
    __shared__ float sv[kWarpsPerBlock];
    __shared__ int   si[kWarpsPerBlock];
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = -INFINITY; int k = 0;
        for (int i = 0; i < kWarpsPerBlock; ++i) if (sv[i] > b) { b = sv[i]; k = si[i]; }
        *out = k;
    }
}

} // namespace

void launch_rms_norm(float* y, const float* x, const float* w, int rows, cudaStream_t s) {
    rms_norm_kernel<<<rows, 256, 0, s>>>(y, x, w);
}

void launch_add(float* acc, const float* x, int n, cudaStream_t s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(acc, x, n);
}

void launch_swiglu(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s) {
    const int n = rows * D;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, gate, up, n);
    launch_quant(dst, y, rows, s);
}

void launch_swiglu_ffn(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s) {
    const int n = rows * Model::FFN;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, gate, up, n);
    launch_quant_ffn(dst, y, rows, s);
}

void launch_sample(const float* logits, int* tok, cudaStream_t s) {
    greedy_kernel<<<1, 256, 0, s>>>(logits, tok);
}

} // namespace nova
