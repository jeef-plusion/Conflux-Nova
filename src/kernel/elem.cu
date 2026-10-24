// nova_v2/src/kernel/elem.cu — 取词向量 / bias / KV 写入
#include "nova/kernel.h"
#include "nova/quant/q4_k.h"

namespace nova {
namespace {

constexpr int kDem = Model::D;

// 一个 block 取一个 token 的整行（每个线程负责每一段 super-block 的同一位置）
__global__ void embed_kernel(float* __restrict__ x, const block_q4_K* __restrict__ emb,
                             const int* __restrict__ ids, int n) {
    const int t = blockIdx.x;
    const int i = threadIdx.x;
    const block_q4_K* row = emb + (int64_t)ids[t] * (kDem / kQK);

    for (int sb = 0; sb < kDem / kQK; ++sb) {
        const block_q4_K* b = row + sb;
        const int g   = i >> 6;
        const int r   = i & 63;
        const int sub = r >> 5;
        const int e   = r & 31;
        uint8_t sc, m;
        get_scale_min_k4(g * 2 + sub, b->scales, sc, m);
        const int byte = b->qs[g * 32 + e];
        const int q = sub ? (byte >> 4) : (byte & 0xF);
        x[(int64_t)t * kDem + sb * kQK + i] =
            half_to_float(b->d) * sc * q - half_to_float(b->dmin) * m;
    }
}

__global__ void bias_kernel(float* __restrict__ y, const float* __restrict__ b, int n, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += b[i % cols];
}

__global__ void kv_store_kernel(float* __restrict__ kc, float* __restrict__ vc,
                                const float* __restrict__ k, const float* __restrict__ v,
                                int64_t pos0, int n) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int64_t pos = pos0 + t;
    const int64_t off = (pos * Model::NKV + h) * Model::HD + d;
    const int64_t src = ((int64_t)t * Model::NKV + h) * Model::HD + d;
    kc[off] = k[src];
    vc[off] = v[src];
}

} // namespace

void launch_embed(float* x, const uint8_t* emb, const int* ids, int n, cudaStream_t s) {
    embed_kernel<<<n, 256, 0, s>>>(x, reinterpret_cast<const block_q4_K*>(emb), ids, n);
}

void launch_bias(float* y, const float* b, int n, int cols, cudaStream_t s) {
    bias_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, b, n, cols);
}

void launch_kv_store(float* kc, float* vc, const float* k, const float* v, int n, int64_t pos0, cudaStream_t s) {
    dim3 grid(n, Model::NKV);
    kv_store_kernel<<<grid, Model::HD, 0, s>>>(kc, vc, k, v, pos0, n);
}

} // namespace nova
