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

// Paged 版：一条 lane 的活与 kv_store_kernel 完全一样，只是目标地址改成
//   pool + ((layer*NB + phys)*KB + slot) * KVW，phys 来自本会话块表。
// 写进去的 float 与旧版逐位相同（同一份 k/v，同一个值），只是落点换了。
//
// kb_shift 由 host 传（= nova::kKvBlockShift，唯一定义在 graph/kv_pool_cfg.h）：
// nvcc 的预处理器按本地代码页解码源文件，那个头里的中文字符串会把收尾引号吞掉，
// 所以 .cu 侧不能 include 它。代价：每 token 一次移位 + 一次掩码，可忽略。
__global__ void kv_store_paged_kernel(float* __restrict__ pool_k, float* __restrict__ pool_v,
                                      const int32_t* __restrict__ table,
                                      int layer, int nb, int kb_shift, int64_t pos0, int n,
                                      const float* __restrict__ k, const float* __restrict__ v) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int64_t pos  = pos0 + t;
    const int64_t kb   = (int64_t)1 << kb_shift;
    const int     mask = (int)(kb - 1);
    const int     phys = __ldg(table + (pos >> kb_shift));
    const int     slot = (int)pos & mask;
    const int64_t off  = (((int64_t)layer * nb + phys) * kb + slot) * (int64_t)Model::NKV * Model::HD
                       + (int64_t)h * Model::HD + d;
    const int64_t src  = ((int64_t)t * Model::NKV + h) * Model::HD + d;
    pool_k[off] = k[src];
    pool_v[off] = v[src];
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

void launch_kv_store_paged(float* pool_k, float* pool_v, const int32_t* table, int layer, int nb,
                           int kb_shift, int n, int64_t pos0, const float* k, const float* v, cudaStream_t s) {
    dim3 grid(n, Model::NKV);
    kv_store_paged_kernel<<<grid, Model::HD, 0, s>>>(pool_k, pool_v, table, layer, nb, kb_shift,
                                                      pos0, n, k, v);
}

} // namespace nova
