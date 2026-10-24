// nova_v2/src/kernel/mmq.cu — Q4_K 的 MMQ 瓦片（prefill 专用）
// 64 行 × 32 token 一块；权重/激活先进 smem；lane 负责整行的一个 32-元素子块，
// 权重只解包一次、喂 4 个 token，把 LSU 从 1 load / 2 dp4a 拉到 1 / 5。
#include "nova/kernel.h"
#include "nova/quant/q4_k.h"

namespace nova {
namespace {

constexpr int BI = 64;    // 行瓦片
constexpr int BJ = 32;    // token 瓦片
constexpr int KC = 256;   // 每次 k 迭代 = 一个 super-block

__global__ void __launch_bounds__(256, 2)
mmq_kernel(const block_q4_K* __restrict__ w, const int8_t* __restrict__ xq,
           const float* __restrict__ xd, float* __restrict__ out, int K, int R, int n_tokens) {
    __shared__ int4   sw[BI * 9];      // 64 行 × 144B
    __shared__ int8_t sx[BJ * KC];     // 32 token × 256B
    __shared__ float  sd[BJ * 8];      // 激活 scale
    __shared__ int    ss[BJ * 8];      // 激活 int8 行和

    const int tx = (int)threadIdx.x;   // 0..31：行
    const int ty = (int)threadIdx.y;   // 0..7 ：token 组
    const int tid = ty * 32 + tx;
    const int row0 = (int)blockIdx.x * BI;
    const int tok0 = (int)blockIdx.y * BJ;

    float acc[2][4] = {};
    const int nsb = K / KC;

    for (int sb = 0; sb < nsb; ++sb) {
        for (int idx = tid; idx < BI * 9; idx += 256) {
            const int r = idx / 9, p = idx - r * 9;
            sw[idx] = reinterpret_cast<const int4*>(w + (size_t)(row0 + r) * nsb + sb)[p];
        }
        for (int idx = tid; idx < BJ * 16; idx += 256) {
            const int t = idx >> 4, p = idx & 15;
            reinterpret_cast<int4*>(sx)[t * 16 + p] =
                reinterpret_cast<const int4*>(xq + (size_t)(tok0 + t) * K + sb * KC)[p];
        }
        __syncthreads();      // sx 由别的线程写入，算 ss 之前必须同步
        {
            const int t = tid >> 3, b = tid & 7;
            const int4* p = reinterpret_cast<const int4*>(sx + t * KC + b * 32);
            int s = 0;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {   // 32 字节 = 2 个 int4 = 8 条 dp4a
                const int4 v = p[i];
                s = __dp4a(v.x, 0x01010101, s);
                s = __dp4a(v.y, 0x01010101, s);
                s = __dp4a(v.z, 0x01010101, s);
                s = __dp4a(v.w, 0x01010101, s);
            }
            ss[t * 8 + b] = s;
            sd[t * 8 + b] = xd[(size_t)(tok0 + t) * (K / 32) + sb * 8 + b];
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int hi = c & 1;
            const int g  = c >> 1;

            int xw[4][8];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int j = ty + 8 * k;
                const int4* px = reinterpret_cast<const int4*>(sx + j * KC + c * 32);
                const int4 a0 = px[0], a1 = px[1];
                xw[k][0] = a0.x; xw[k][1] = a0.y; xw[k][2] = a0.z; xw[k][3] = a0.w;
                xw[k][4] = a1.x; xw[k][5] = a1.y; xw[k][6] = a1.z; xw[k][7] = a1.w;
            }

            #pragma unroll
            for (int r2 = 0; r2 < 2; ++r2) {
                const int i = tx + r2 * 32;
                const int4* row = sw + i * 9;
                uint8_t sc, m;
                get_scale_min_k4(c, reinterpret_cast<const uint8_t*>(row) + 4, sc, m);
                const float d  = half_to_float(((const uint16_t*)row)[0]) * (float)sc;
                const float dm = half_to_float(((const uint16_t*)row)[1]) * (float)m;

                const int4 w0 = row[1 + 2 * g], w1 = row[2 + 2 * g];
                const unsigned words[8] = {(unsigned)w0.x, (unsigned)w0.y, (unsigned)w0.z, (unsigned)w0.w,
                                           (unsigned)w1.x, (unsigned)w1.y, (unsigned)w1.z, (unsigned)w1.w};
                int q[8];
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    unsigned v = words[t];
                    if (hi) v >>= 4;
                    v &= 0x0F0F0F0Fu;
                    q[t] = (int)((v & 0xF) | ((v >> 8 & 0xF) << 8) | ((v >> 16 & 0xF) << 16) | ((v >> 24 & 0xF) << 24));
                }

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int j = ty + 8 * k;
                    int dot = 0;
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) dot = __dp4a(q[t], xw[k][t], dot);
                    const float dx = sd[j * 8 + c];
                    const float sxv = (float)ss[j * 8 + c];
                    acc[r2][k] = fmaf(dx, fmaf(d, (float)dot, -dm * sxv), acc[r2][k]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r2 = 0; r2 < 2; ++r2) {
        const int i = row0 + tx + r2 * 32;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = tok0 + ty + 8 * k;
            if (j < n_tokens) out[(size_t)j * R + i] = acc[r2][k];
        }
    }
}

} // namespace

template <int K, int R>
void mmq_launch(const Q4KWeight& w, const ActQ8& x, float* o, int n, cudaStream_t s) {
    static_assert(K % KC == 0, "K must be a multiple of 256");
    static_assert(R % BI == 0, "R must be a multiple of 64");
    dim3 grid((unsigned)(R / BI), (unsigned)((n + BJ - 1) / BJ));
    mmq_kernel<<<grid, dim3(32, 8), 0, s>>>(
        reinterpret_cast<const block_q4_K*>(w.blocks), x.qs, x.d, o, K, R, n);
}

#define NOVA_INST_MMQ(K, R) template void mmq_launch<K, R>(const Q4KWeight&, const ActQ8&, float*, int, cudaStream_t);

NOVA_INST_MMQ(Model::D,         Model::NH * Model::HD)
NOVA_INST_MMQ(Model::D,         Model::NKV * Model::HD)
NOVA_INST_MMQ(Model::NH * Model::HD, Model::D)
NOVA_INST_MMQ(Model::D,         Model::FFN)
NOVA_INST_MMQ(Model::FFN,       Model::D)
NOVA_INST_MMQ(Model::D,         Model::V)

} // namespace nova
