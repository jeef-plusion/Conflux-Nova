// nova_v2/src/kernel/sycl/moe.cpp — DeepSeek-V2-Lite 的 MoE 前段（路由 + top-k）
//
// 一个 MoE 层（V2-Lite 的 26 层 MoE）在 decode 时要做的事：
//   1. scores = sigmoid(W_router · x)              —— 64 个 routed expert 的打分
//   2. 取分数最高的 6 个（top-6），按这 6 个的和归一化 —— 得到 6 个权重
//   3. 对每个选中的 expert：y += w_e * down_e( swiglu(gate_e(x) * up_e(x)) )
//   4. 再加 2 个 shared expert（恒开、权重 1）：y += down_s( swiglu(gate_s(x) * up_s(x)) )
//
// 本文件只负责 1 和 2（两个小 kernel，可单独对拍）；第 3、4 步是「按 idx 逐个调 gemv」，
// 由 runner 驱动（专家权重是 [nExpert][expertFFN][D] 的 Q4_K，行偏移算好即可）。
//
// 与 HF 的对应：DeepSeek-V2 的 config 里 scoring_func='sigmoid'、topk_method='greedy'、
// norm_topk_prob=true（V3 才引入 group-limited routing，这里不涉及）。
#include "kernels.h"

#include "nova/model/deepseek_v2_lite.h"

namespace nova {
namespace syclk {
namespace {

using DSV2 = nova::DeepSeekV2Lite;

static_assert(DSV2::kNExpert <= 1024, "router kernel 假设 nExpert 能放进一个 WG");

// ---------------------------------------------------------------------------
// 打分：一个 work-item 管一个 expert，做 D 维点积后过 sigmoid
// ---------------------------------------------------------------------------
class MoeRouterKernel {
public:
    MoeRouterKernel(const float* x, const float* w, float* out, int n_expert, int d)
        : x_(x), w_(w), out_(out), n_exp_(n_expert), d_(d) {}

    void operator()(sycl::nd_item<1> it) const {
        const int e = static_cast<int>(it.get_global_linear_id());
        if (e >= n_exp_) return;
        const float* wr = w_ + static_cast<std::int64_t>(e) * d_;
        float acc = 0.f;
        for (int i = 0; i < d_; ++i) acc = sycl::fma(x_[i], wr[i], acc);
        out_[e] = 1.f / (1.f + sycl::exp(-acc));     // sigmoid
    }

private:
    const float* x_;
    const float* w_;
    float*       out_;
    int          n_exp_;
    int          d_;
};

// ---------------------------------------------------------------------------
// top-k + 归一化：单 work-item 选择排序（64 个元素，代价可忽略）
// ---------------------------------------------------------------------------
class MoeTopKKernel {
public:
    MoeTopKKernel(const float* scores, int* idx, float* w, int n_expert, int topk)
        : s_(scores), idx_(idx), w_(w), n_exp_(n_expert), topk_(topk) {}

    void operator()(sycl::nd_item<1> it) const {
        if (it.get_global_linear_id() != 0) return;

        float best_w[DSV2::kTopK];
        int   best_i[DSV2::kTopK];
        bool  taken[DSV2::kNExpert];
        for (int e = 0; e < n_exp_; ++e) taken[e] = false;

        for (int k = 0; k < topk_; ++k) {
            int bi = -1;
            float bs = -INFINITY;
            for (int e = 0; e < n_exp_; ++e) {
                if (taken[e]) continue;
                if (s_[e] > bs) { bs = s_[e]; bi = e; }
            }
            taken[bi] = true;                 // n_exp_ >= topk_ 保证 bi 有效
            best_i[k] = bi;
            best_w[k] = bs;
        }

        // norm_topk_prob：按选中项之和归一化
        float sum = 0.f;
        for (int k = 0; k < topk_; ++k) sum += best_w[k];
        const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
        for (int k = 0; k < topk_; ++k) {
            idx_[k] = best_i[k];
            w_[k]   = best_w[k] * inv;
        }
    }

private:
    const float* s_;
    int*         idx_;
    float*       w_;
    int          n_exp_;
    int          topk_;
};

} // namespace

void moe_router(const float* x, const float* w_router, float* scores, sycl::queue& q) {
    q.parallel_for<MoeRouterKernel>(
        sycl::nd_range<1>(sycl::range<1>(DSV2::kNExpert), sycl::range<1>(DSV2::kNExpert)),
        MoeRouterKernel(x, w_router, scores, DSV2::kNExpert, DSV2::D));
}

void moe_topk(const float* scores, int* idx, float* w, sycl::queue& q) {
    // 32 个 work-item、只有 0 号干活：一个 work-group 小于 subgroup 时，
    // 有的实现在 reqd_sub_group_size 上会挑刺，干脆给满一个 subgroup。
    q.parallel_for<MoeTopKKernel>(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
                                  MoeTopKKernel(scores, idx, w, DSV2::kNExpert, DSV2::kTopK));
}

} // namespace syclk
} // namespace nova
