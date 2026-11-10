// nova_v2/src/kernel/sycl/mla.cpp — DeepSeek-V2-Lite 的 Multi-head Latent Attention（单 token decode）
//
// MLA 与 attn.cpp 的 attn_decode 是**两种不同的注意力**，不能互相替代：
//
//   attn_decode（Qwen/Llama）每个 KV head 的 K 是 HD=128 维、V 也是 128 维，K/V 直接进 cache。
//   MLA（DeepSeek-V2）      每个 head 的 K 是 [nope 128 | rope 64] = 192 维，V 是 128 维；
//                           cache 里本应只存压缩态 kv_a（512 + 64 = 576 / token / 层），
//                           用的时候再由 kv_b 展开。
//
// 本文件提供三件事（都在单 token decode 口径下）：
//   mla_rope_apply : 只旋转 q/k 的最后 64 维（Neox 前后半对）
//   mla_kv_store   : 把「展开后的 k_nope / k_rope / v」写进 cache 的第 pos 行
//   mla_decode     : 192 维 key、128 维 value 的 causal 注意力
//
// cache 布局（per layer，展开后）：
//   kc : [ctx][NH][192]   K（rope 已旋转）
//   vc : [ctx][NH][128]   V
//   qv : [NH][192]        当前 token 的 q（nope 由 q_proj 给出，rope 部分已旋转）
//   o  : [NH][128]
//
// 为什么不只存压缩态、读时再展开：那要在注意力内层对每个位置做 2 次 512×4096 的 GEMV，
// 27 层 × 512 位置 × ~1.2 MB，代价完全不可接受。展开一次存下来（ctx=512 时 ~283 MB）换取
// 内层只做点积。要省这块显存就上 matrix-absorbed（把 kv_b 折进 q/o 权重），那是下一步。
#include "kernels.h"

#include "nova/model/deepseek_v2_lite.h"

namespace nova {
namespace syclk {
namespace {

using DSV2 = nova::DeepSeekV2Lite;

inline constexpr int kKeyW  = DSV2::kQPerHead;                        // 192
inline constexpr int kValW  = DSV2::kVPerHead;                        // 128
inline constexpr int kSg    = syclb::kSyclSgAttn;                     // 32
inline constexpr int kNDqk  = kKeyW / kSg;                            // 6
inline constexpr int kNDv   = kValW / kSg;                            // 4
inline constexpr int kWG    = 128;                                    // 每 head 的 WG 大小
inline constexpr int kWarp  = kWG / kSg;                              // 4 个 subgroup
inline constexpr int kSd    = 2 + kValW;                              // local 里每个 subgroup 的跨度
inline constexpr float kScale = 1.f / 1.4142135623730951f / 9.797958971f;  // 1/sqrt(192) 的近似
inline constexpr float kAttnScale = 0.07216878364870322f;             // 1/sqrt(192) 精确值

static_assert(kKeyW % kSg == 0, "MLA: q/k 宽度必须能被 32 整除");
static_assert(kValW % kSg == 0, "MLA: v 宽度必须能被 32 整除");
static_assert(kKeyW <= 256, "MLA: q 进 local 的两轮装载假设 kKeyW <= 2*WG");

// ---------------------------------------------------------------------------
// rope：只转 q/k 的 rope 部分（每 head 64 维，Neox 前后半对）
// ---------------------------------------------------------------------------
class MlaRopeKernel {
public:
    MlaRopeKernel(float* q, float* k, const float* cs, const float* sn, std::int64_t pos)
        : q_(q), k_(k), cs_(cs), sn_(sn), pos_(pos) {}

    void operator()(sycl::nd_item<1> it) const {
        constexpr int kHalf = DSV2::kQKRopeDim / 2;            // 32 对
        const int gid  = static_cast<int>(it.get_global_linear_id());
        const int head = gid / kHalf;
        const int i    = gid - head * kHalf;

        // 64 维 rope 第 i 对用 theta^(-2i/64)；rope_table 存的是 theta^(-j/64)（128 维口径），
        // 所以取 j = 2i 那一条。
        const float c = cs_[pos_ * 2 * kHalf + 2 * i];
        const float s = sn_[pos_ * 2 * kHalf + 2 * i];

        float* base = (head < DSV2::NH)
                          ? q_ + static_cast<std::int64_t>(head) * kKeyW + DSV2::kQKNopeDim
                          : k_ + static_cast<std::int64_t>(head - DSV2::NH) * DSV2::kQKRopeDim;
        const float x0 = base[i];
        const float x1 = base[i + kHalf];
        base[i]         = x0 * c - x1 * s;
        base[i + kHalf] = x0 * s + x1 * c;
    }

private:
    float*       q_;
    float*       k_;
    const float* cs_;
    const float* sn_;
    std::int64_t pos_;
};

// ---------------------------------------------------------------------------
// kv_store：写 cache 第 pos 行
//   kn : [NH][128]  k_nope（kv_b 的前半输出）
//   kr : [NH][64]   k_rope（= kv_a 的最后 64 维，rope 已旋转）
//   v  : [NH][128]  v（kv_b 的后半输出）
// ---------------------------------------------------------------------------
class MlaKvStoreKernel {
public:
    MlaKvStoreKernel(float* kc, float* vc, const float* kn, const float* kr, const float* v,
                     std::int64_t pos)
        : kc_(kc), vc_(vc), kn_(kn), kr_(kr), v_(v), pos_(pos) {}

    void operator()(sycl::nd_item<2> it) const {
        const int head = static_cast<int>(it.get_group(0));
        const int d    = static_cast<int>(it.get_local_id(1));
        float* krow = kc_ + pos_ * DSV2::NH * kKeyW + static_cast<std::int64_t>(head) * kKeyW;
        if (d < DSV2::kQKNopeDim) {
            krow[d] = kn_[head * DSV2::kQKNopeDim + d];
        } else {
            krow[d] = kr_[head * DSV2::kQKRopeDim + (d - DSV2::kQKNopeDim)];
        }
        if (d < kValW) {
            float* vrow = vc_ + pos_ * DSV2::NH * kValW + static_cast<std::int64_t>(head) * kValW;
            vrow[d] = v_[head * kValW + d];
        }
    }

private:
    float*       kc_;
    float*       vc_;
    const float* kn_;
    const float* kr_;
    const float* v_;
    std::int64_t pos_;
};

// ---------------------------------------------------------------------------
// decode：一个 WG 管一个 head，4 个 subgroup 各扫一段 key，local 合并 (m, l, acc)
// ---------------------------------------------------------------------------
class MlaDecodeKernel {
public:
    MlaDecodeKernel(const float* q, const float* kc, const float* vc, float* o, int n_keys,
                    sycl::local_accessor<float, 1> sq, sycl::local_accessor<float, 1> sp)
        : q_(q), kc_(kc), vc_(vc), o_(o), n_keys_(n_keys), sq_(sq), sp_(sp) {}

    void operator()(sycl::nd_item<1> it) const {
        const int head = static_cast<int>(it.get_group(0));
        const int tid  = static_cast<int>(it.get_local_id(0));
        const int sgid = tid / kSg;
        const int lane = tid % kSg;
        const sycl::sub_group sg = it.get_sub_group();

        // q 进 local（192 个，两轮装载）
        for (int i = tid; i < kKeyW; i += kWG) sq_[i] = q_[head * kKeyW + i];
        it.barrier(sycl::access::fence_space::local_space);

        const int chunk = (n_keys_ + kWarp - 1) / kWarp;
        const int kv0   = sgid * chunk;
        int kv1         = kv0 + chunk;
        if (kv1 > n_keys_) kv1 = n_keys_;

        float m = -INFINITY;
        float l = 0.f;
        float acc[kNDv] = {};

        for (int p = kv0; p < kv1; ++p) {
            const float* kp = kc_ + static_cast<std::int64_t>(p) * DSV2::NH * kKeyW +
                              static_cast<std::int64_t>(head) * kKeyW;
            float part = 0.f;
            #pragma unroll
            for (int c = 0; c < kNDqk; ++c) {
                part = sycl::fma(sq_[lane + kSg * c], kp[lane + kSg * c], part);
            }
            const float qk = sg_sum<float>(sg, part) * kAttnScale;

            const float mn   = sycl::fmax(m, qk);
            const float corr = sycl::exp(m - mn);
            const float pr   = sycl::exp(qk - mn);
            l = sycl::fma(l, corr, pr);

            const float* vp = vc_ + static_cast<std::int64_t>(p) * DSV2::NH * kValW +
                              static_cast<std::int64_t>(head) * kValW;
            #pragma unroll
            for (int c = 0; c < kNDv; ++c) {
                acc[c] = sycl::fma(acc[c], corr, pr * vp[lane + kSg * c]);
            }
            m = mn;
        }

        // 4 个 subgroup 在 local 里合并（与 attn.cpp 的写法一致）
        if (lane == 0) {
            sp_[sgid * kSd + 0] = m;
            sp_[sgid * kSd + 1] = l;
        }
        #pragma unroll
        for (int c = 0; c < kNDv; ++c) sp_[sgid * kSd + 2 + lane + kSg * c] = acc[c];
        it.barrier(sycl::access::fence_space::local_space);

        if (sgid == 0) {
            float mm = -INFINITY;
            #pragma unroll
            for (int w = 0; w < kWarp; ++w) mm = sycl::fmax(mm, sp_[w * kSd + 0]);
            float ll = 0.f;
            #pragma unroll
            for (int w = 0; w < kWarp; ++w) {
                const float ls = sp_[w * kSd + 1];
                if (ls > 0.f) ll = sycl::fma(ls, sycl::exp(sp_[w * kSd + 0] - mm), ll);
            }
            const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;
            #pragma unroll
            for (int c = 0; c < kNDv; ++c) {
                const int dc = 2 + lane + kSg * c;
                float a = 0.f;
                #pragma unroll
                for (int w = 0; w < kWarp; ++w) {
                    a = sycl::fma(sp_[w * kSd + dc], sycl::exp(sp_[w * kSd + 0] - mm), a);
                }
                o_[head * kValW + lane + kSg * c] = a * inv;
            }
        }
    }

private:
    const float* q_;
    const float* kc_;
    const float* vc_;
    float*       o_;
    int          n_keys_;
    sycl::local_accessor<float, 1> sq_;
    sycl::local_accessor<float, 1> sp_;
};

} // namespace

void mla_rope_apply(float* qv, float* kv, std::int64_t pos, const float* cs, const float* sn,
                    sycl::queue& q) {
    constexpr int kHalf = DSV2::kQKRopeDim / 2;
    q.parallel_for<MlaRopeKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(DSV2::NH) * 2 * kHalf),
                          sycl::range<1>(kHalf)),
        MlaRopeKernel(qv, kv, cs, sn, pos));
}

void mla_kv_store(float* kc, float* vc, const float* kn, const float* kr, const float* v,
                  std::int64_t pos, sycl::queue& q) {
    q.parallel_for<MlaKvStoreKernel>(
        sycl::nd_range<2>(sycl::range<2>(DSV2::NH, kKeyW), sycl::range<2>(1, kKeyW)),
        MlaKvStoreKernel(kc, vc, kn, kr, v, pos));
}

void mla_decode(const float* qv, const float* kc, const float* vc, float* o, int n_past,
                sycl::queue& q) {
    const int n_keys = n_past + 1;
    if (n_keys <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(kKeyW), h);
        sycl::local_accessor<float, 1> sp(sycl::range<1>(kWarp * kSd), h);
        h.parallel_for<MlaDecodeKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(DSV2::NH) * kWG),
                              sycl::range<1>(kWG)),
            MlaDecodeKernel(qv, kc, vc, o, n_keys, sq, sp));
    });
}

} // namespace syclk
} // namespace nova
