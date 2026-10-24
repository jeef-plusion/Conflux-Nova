// nova_v2/src/kernel/sycl/attn.cpp — decode attention（split-K）+ prefill attention（tile）
//
// decode  : 与 src/kernel/attn.cu 的 fa_split_kernel / fa_combine_kernel 同构。
//           一个 WorkGroup = 4 个 subgroup，各扫自己那段 KV，block 内在 local 里合并，
//           再写 partial[head][split][m,l,acc]，最后由 combine kernel 归一化。
// prefill : CUDA 版是"1 warp 1 个 (head,q)"，每个 q 把整段 KV 流一遍 —— 预填充阶段
//           KV 流量 = n_q * NH * n_past * HD * 2 * 4B，512 token 时约 7.5 GB，不可接受。
//           这里改成 tile：一个 WorkGroup 管 (head, 8 个 query)，KV 按 16 个 key 一批
//           搬进 local，8 个 query 共享一次搬运 → KV 流量 /8。
//           对照 ggml-sycl/fattn-tile.hpp : flash_attn_tile()（它按 nbatch_K=64、
//           nbatch_fa=64 的 tile 走，我们受 32 KiB local 限制只能 16 x 128 x 2）。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

// 1/sqrt(HD)：与 CUDA 版共用一个常量表达式，保证两条路径逐位可比
template <int D>
inline constexpr float attn_scale_of() {
    return (D == 64) ? 0.125f : 0.08838834764831845f;
}
inline constexpr float kAttnScale = attn_scale_of<kHD>();

inline constexpr int kND      = kHD / 32;               // 每 lane 持几个 float
inline constexpr int kWarpDec = kAttnDecWg / kSyclSg;   // decode 的 subgroup 数 = 4
inline constexpr int kSd      = kHD + 2;                // partial 行宽：m, l, HD
inline constexpr int kPreWG   = kPrefillTQ * kSyclSg;
inline constexpr int kKvTile  = kPrefillKC * kHD;       // K 与 V 各自的 tile 元素数

} // namespace

// kernel 名标签（lambda 走 parallel_for<Name>，Name 必须是完整类型）
struct FaSplitKernel {};
struct FaCombineKernel {};
struct FaPrefillKernel {};

// ===========================================================================
// decode: split-K
// ===========================================================================
void attn_decode(const float* qv, const float* kc, const float* vc, float* o,
                 int n_past, float* scratch, sycl::queue& q) {
    const int n_keys = n_past + 1;
    if (n_keys <= 0) return;

    int nsplit = (n_keys + 31) / 32;
    if (nsplit > kAttnSplit) nsplit = kAttnSplit;
    if (nsplit < 1) nsplit = 1;
    const int chunk = (n_keys + nsplit - 1) / nsplit;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(kHD), h);
        sycl::local_accessor<float, 1> sp(sycl::range<1>(kWarpDec * kSd), h);
        h.parallel_for<FaSplitKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(kNH) * nsplit * kAttnDecWg),
                              sycl::range<1>(kAttnDecWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSyclSg)]] {
                const int tid  = static_cast<int>(it.get_local_id(0));
                const int sgid = tid / kSyclSg;
                const int lane = tid % kSyclSg;
                const int gid  = static_cast<int>(it.get_group(0));
                const int head = gid / nsplit;
                const int spi  = gid - head * nsplit;
                const int kvh  = head / (kNH / kNKV);

                const int q4n = (chunk + kWarpDec - 1) / kWarpDec;
                const int kv0 = spi * chunk + sgid * q4n;
                int kv1 = kv0 + q4n;
                const int lim = (spi + 1) * chunk;
                if (kv1 > n_keys) kv1 = n_keys;
                if (kv1 > lim) kv1 = lim;

                if (tid < kHD) sq[tid] = qv[head * kHD + tid];
                it.barrier(sycl::access::fence_space::local_space);

                float m = -INFINITY;
                float l = 0.f;
                float acc[kND] = {};
                const float* qh = sq;

                for (int p = kv0; p < kv1; ++p) {
                    const float* kp = kc + (static_cast<std::int64_t>(p) * kNKV + kvh) * kHD;
                    float part = 0.f;
                    #pragma unroll
                    for (int c = 0; c < kND; ++c) {
                        part = sycl::fma(qh[lane + 32 * c], kp[lane + 32 * c], part);
                    }
                    const float qk = sg_sum<float>(it.get_sub_group(), part) * kAttnScale;

                    const float mn   = sycl::fmax(m, qk);
                    const float corr = sycl::exp(m - mn);
                    const float pr   = sycl::exp(qk - mn);
                    l = sycl::fma(l, corr, pr);

                    const float* vp = vc + (static_cast<std::int64_t>(p) * kNKV + kvh) * kHD;
                    #pragma unroll
                    for (int c = 0; c < kND; ++c) {
                        acc[c] = sycl::fma(acc[c], corr, pr * vp[lane + 32 * c]);
                    }
                    m = mn;
                }

                // block 内 4 个 subgroup 先在 local 里合并，避免把 4 份 partial 写回显存
                if (lane == 0) {
                    sp[sgid * kSd + 0] = m;
                    sp[sgid * kSd + 1] = l;
                }
                #pragma unroll
                for (int c = 0; c < kND; ++c) sp[sgid * kSd + 2 + lane + 32 * c] = acc[c];
                it.barrier(sycl::access::fence_space::local_space);

                if (sgid == 0) {
                    float mm = -INFINITY;
                    #pragma unroll
                    for (int w = 0; w < kWarpDec; ++w) mm = sycl::fmax(mm, sp[w * kSd + 0]);
                    float ll = 0.f;
                    #pragma unroll
                    for (int w = 0; w < kWarpDec; ++w) {
                        const float ls = sp[w * kSd + 1];
                        if (ls > 0.f) ll = sycl::fma(ls, sycl::exp(sp[w * kSd + 0] - mm), ll);
                    }
                    float* out = scratch + (static_cast<std::size_t>(head) * nsplit + spi) * kSd;
                    if (lane == 0) {
                        out[0] = mm;
                        out[1] = ll;
                    }
                    #pragma unroll
                    for (int c = 0; c < kND; ++c) {
                        const int dc = 2 + lane + 32 * c;
                        float a = 0.f;
                        #pragma unroll
                        for (int w = 0; w < kWarpDec; ++w) {
                            a = sycl::fma(sp[w * kSd + dc], sycl::exp(sp[w * kSd + 0] - mm), a);
                        }
                        out[dc] = a;
                    }
                }
            });
    });

    q.parallel_for<FaCombineKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(kNH) * kHD), sycl::range<1>(kHD)),
        [=](sycl::nd_item<1> it) {
            const int head = static_cast<int>(it.get_group(0));
            const int d    = static_cast<int>(it.get_local_id(0));
            const float* base = scratch + static_cast<std::size_t>(head) * nsplit * kSd;

            float m = -INFINITY;
            for (int s = 0; s < nsplit; ++s) m = sycl::fmax(m, base[s * kSd + 0]);
            float l = 0.f;
            for (int s = 0; s < nsplit; ++s) {
                const float ls = base[s * kSd + 1];
                if (ls > 0.f) l = sycl::fma(ls, sycl::exp(base[s * kSd + 0] - m), l);
            }
            float a = 0.f;
            for (int s = 0; s < nsplit; ++s) {
                a = sycl::fma(base[s * kSd + 2 + d], sycl::exp(base[s * kSd + 0] - m), a);
            }
            o[head * kHD + d] = (l > 0.f) ? a / l : 0.f;
        });
}

// ===========================================================================
// prefill: tile（query 方向 8 个一组，KV 方向 16 个一批进 local）
// ===========================================================================
void attn_prefill(const float* qv, const float* kc, const float* vc, float* o,
                  int n_q, int n_past, sycl::queue& q) {
    if (n_q <= 0) return;
    const int n_keys = n_past + n_q;
    const int n_tiles = div_up(n_q, kPrefillTQ);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> skv(sycl::range<1>(2 * kKvTile), h);
        h.parallel_for<FaPrefillKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(kNH) * n_tiles * kPreWG),
                              sycl::range<1>(kPreWG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSyclSg)]] {
                const int tid  = static_cast<int>(it.get_local_id(0));
                const int sgid = tid / kSyclSg;
                const int lane = tid % kSyclSg;
                const sycl::sub_group sg = it.get_sub_group();

                const int gid  = static_cast<int>(it.get_group(0));
                const int head = gid % kNH;
                const int tile = gid / kNH;
                const int qidx = tile * kPrefillTQ + sgid;
                const bool active = (qidx < n_q);
                const std::int64_t pos = static_cast<std::int64_t>(n_past) + qidx;
                const int kvh = head / (kNH / kNKV);

                float qreg[kND] = {};
                if (active) {
                    const float* qp = qv + (static_cast<std::int64_t>(qidx) * kNH + head) * kHD;
                    #pragma unroll
                    for (int c = 0; c < kND; ++c) qreg[c] = qp[lane + 32 * c];
                }

                float m = -INFINITY;
                float l = 0.f;
                float acc[kND] = {};

                for (int k0 = 0; k0 < n_keys; k0 += kPrefillKC) {
                    // 协作搬运：K 与 V 各 kKvTile 个 float，kPreWG 个 work-item 平分
                    #pragma unroll
                    for (int s = 0; s < div_up(kKvTile, kPreWG); ++s) {
                        const int flat = tid + s * kPreWG;
                        if (flat < kKvTile) {
                            const int krow = flat / kHD;
                            const int d    = flat - krow * kHD;
                            const int p    = k0 + krow;
                            if (p < n_keys) {
                                const std::int64_t off = (static_cast<std::int64_t>(p) * kNKV + kvh) * kHD + d;
                                skv[flat]            = kc[off];
                                skv[kKvTile + flat]  = vc[off];
                            } else {
                                skv[flat]           = 0.f;
                                skv[kKvTile + flat] = 0.f;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (active) {
                        #pragma unroll 1
                        for (int kk = 0; kk < kPrefillKC; ++kk) {
                            const std::int64_t p = static_cast<std::int64_t>(k0) + kk;
                            if (p > pos || p >= n_keys) break;
                            float part = 0.f;
                            #pragma unroll
                            for (int c = 0; c < kND; ++c) {
                                part = sycl::fma(qreg[c], skv[kk * kHD + lane + 32 * c], part);
                            }
                            const float qk = sg_sum<float>(sg, part) * kAttnScale;

                            const float mn   = sycl::fmax(m, qk);
                            const float corr = sycl::exp(m - mn);
                            const float pr   = sycl::exp(qk - mn);
                            l = sycl::fma(l, corr, pr);
                            #pragma unroll
                            for (int c = 0; c < kND; ++c) {
                                acc[c] = sycl::fma(acc[c], corr, pr * skv[kKvTile + kk * kHD + lane + 32 * c]);
                            }
                            m = mn;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (active) {
                    float* op = o + (static_cast<std::int64_t>(qidx) * kNH + head) * kHD;
                    const float inv = (l > 0.f) ? 1.f / l : 0.f;
                    #pragma unroll
                    for (int c = 0; c < kND; ++c) op[lane + 32 * c] = acc[c] * inv;
                }
            });
    });
}

} // namespace syclk
} // namespace nova
