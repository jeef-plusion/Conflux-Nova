// nova_v2/src/kernel/sycl/rope.cpp — RoPE（NeoX / 交错两种布局，编译期二选一）
//
// 布局选择与 CUDA 版完全一致：ArchTraits<Qwen2>::kRope == Neox，ArchTraits<Llama>::kRope == Norm。
// SYCL 版把这件事收敛成一个宏 NOVA_SYCL_ROPE_NEOX（默认 1 = Qwen2 系），只实例化被选中的那份。
//
// 对照 ggml-sycl/rope.cpp : rope_neox() / rope_norm()
//   llama.cpp 的这两个 kernel 把"每一对 (i0, i0+1)"当成一个 work-item（i0 = 2*idx），
//   支持 YaRN / freq_factors / mrope 分区，还要处理 n_offs 与 set_rows。我们只需要
//   最朴素的"基点 pow(theta, -2i/D)"版本，所以每个 work-item 直接处理一对元素：
//     Neox  : (i, i + HD/2)
//     Norm  : (2i, 2i+1)
// 表 (cos/sin) 在显存里，由 rope_table 一次算好（kCtx * HD/2 对），与 CUDA 版同一布局。
#include "kernels.h"

#include <cmath>

namespace nova {
namespace syclk {
namespace {

inline constexpr int kHalf  = kHD / 2;
inline constexpr int kHeads = kNH + kNKV;

class RopeTableKernel {
public:
    RopeTableKernel(float* cs, float* sn, int total) : cs_(cs), sn_(sn), total_(total) {}

    void operator()(sycl::nd_item<1> it) const {
        const int idx = static_cast<int>(it.get_global_linear_id());
        if (idx >= total_) return;
        const int pos = idx / kHalf;
        const int i   = idx - pos * kHalf;
        const float f = sycl::exp2(-static_cast<float>(i) * (1.f / static_cast<float>(kHalf)) *
                                   sycl::log2(static_cast<float>(kTheta)));
        const float a = static_cast<float>(pos) * f;
        cs_[idx] = sycl::cos(a);
        sn_[idx] = sycl::sin(a);
    }

private:
    float* cs_;
    float* sn_;
    int    total_;
};

// 单 token（decode）：group = head，local = i ∈ [0, HD/2)
template <bool NEOX>
class RopeApplyKernel {
public:
    RopeApplyKernel(float* qv, float* kv, const float* cs, const float* sn, std::int64_t pos)
        : q_(qv), k_(kv), cs_(cs), sn_(sn), pos_(pos) {}

    void operator()(sycl::nd_item<1> it) const {
        const int head = static_cast<int>(it.get_group(0));
        const int i    = static_cast<int>(it.get_local_id(0));
        const float c  = cs_[pos_ * kHalf + i];
        const float s  = sn_[pos_ * kHalf + i];

        float* base = (head < kNH) ? q_ + static_cast<std::int64_t>(head) * kHD
                                   : k_ + static_cast<std::int64_t>(head - kNH) * kHD;
        const int a = NEOX ? i : (2 * i);
        const int b = NEOX ? (i + kHalf) : (2 * i + 1);

        const float x0 = base[a];
        const float x1 = base[b];
        base[a] = x0 * c - x1 * s;
        base[b] = x0 * s + x1 * c;
    }

private:
    float*       q_;
    float*       k_;
    const float* cs_;
    const float* sn_;
    std::int64_t pos_;
};

// 批量（prefill）：group = (token, head)
template <bool NEOX>
class RopeBatchKernel {
public:
    RopeBatchKernel(float* qv, float* kv, const float* cs, const float* sn, std::int64_t pos0)
        : q_(qv), k_(kv), cs_(cs), sn_(sn), pos0_(pos0) {}

    void operator()(sycl::nd_item<1> it) const {
        const int gid  = static_cast<int>(it.get_group(0));
        const int t    = gid / kHeads;
        const int head = gid - t * kHeads;
        const int i    = static_cast<int>(it.get_local_id(0));
        const std::int64_t pos = pos0_ + t;

        const float c = cs_[pos * kHalf + i];
        const float s = sn_[pos * kHalf + i];

        float* base = (head < kNH)
                          ? q_ + (static_cast<std::int64_t>(t) * kNH + head) * kHD
                          : k_ + (static_cast<std::int64_t>(t) * kNKV + (head - kNH)) * kHD;
        const int a = NEOX ? i : (2 * i);
        const int b = NEOX ? (i + kHalf) : (2 * i + 1);

        const float x0 = base[a];
        const float x1 = base[b];
        base[a] = x0 * c - x1 * s;
        base[b] = x0 * s + x1 * c;
    }

private:
    float*       q_;
    float*       k_;
    const float* cs_;
    const float* sn_;
    std::int64_t pos0_;
};

} // namespace

void rope_table(float* cs, float* sn, sycl::queue& q) {
    const int total = nova::kCtx * kHalf;
    const int groups = div_up(total, 256);
    q.parallel_for<RopeTableKernel>(sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups) * 256),
                                                      sycl::range<1>(256)),
                                    RopeTableKernel(cs, sn, total));
}

void rope_apply(float* qv, float* kv, std::int64_t pos, const float* cs, const float* sn, sycl::queue& q) {
    if constexpr (kRopeNeox) {
        q.parallel_for<RopeApplyKernel<true>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(kHeads) * kHalf),
                              sycl::range<1>(kHalf)),
            RopeApplyKernel<true>(qv, kv, cs, sn, pos));
    } else {
        q.parallel_for<RopeApplyKernel<false>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(kHeads) * kHalf),
                              sycl::range<1>(kHalf)),
            RopeApplyKernel<false>(qv, kv, cs, sn, pos));
    }
}

void rope_apply_batch(float* qv, float* kv, int n, std::int64_t pos0,
                      const float* cs, const float* sn, sycl::queue& q) {
    if (n <= 0) return;
    const std::size_t groups = static_cast<std::size_t>(n) * kHeads;
    if constexpr (kRopeNeox) {
        q.parallel_for<RopeBatchKernel<true>>(
            sycl::nd_range<1>(sycl::range<1>(groups * kHalf), sycl::range<1>(kHalf)),
            RopeBatchKernel<true>(qv, kv, cs, sn, pos0));
    } else {
        q.parallel_for<RopeBatchKernel<false>>(
            sycl::nd_range<1>(sycl::range<1>(groups * kHalf), sycl::range<1>(kHalf)),
            RopeBatchKernel<false>(qv, kv, cs, sn, pos0));
    }
}

} // namespace syclk
} // namespace nova
