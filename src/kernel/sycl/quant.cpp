// nova_v2/src/kernel/sycl/quant.cpp — fp32 -> int8（每 32 元素一个 fp32 scale，每 16 一个 int 和）
//
// 这就是 ActQ8 的生产者：qs 每元素 1 B，d 每 32 元素 4 B，s 每 16 元素 4 B。
// 与 CUDA 的 quant_kernel 语义逐位一致（同样的 amax/127 缩放、同样的 __float2int_rn
// -> sycl::rint、同样的"每 16 元素求和"）。
//
// 布局选择：**一个 subgroup 吃一个 32 元素块**，lane 各持 1 个 float。
// CUDA 版是一个 thread 吃 32 个元素（跨步 128 B 的访问模式）；这里 32 个 lane
// 合并成一条 128 B 的合并访问，是 iGPU 上更省的做法。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

template <int K>
class QuantKernel {
public:
    QuantKernel(std::int8_t* qs, float* d, std::int32_t* s, const float* x, int total_blocks)
        : qs_(qs), d_(d), s_(s), x_(x), total_(total_blocks) {}

    void operator()(sycl::nd_item<1> it) const [[sycl::reqd_sub_group_size(kSyclSg)]] {
        constexpr int NB = K / 32;
        const int blk = static_cast<int>(it.get_global_linear_id() / kSyclSg);
        if (blk >= total_) return;

        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int row  = blk / NB;
        const int b    = blk - row * NB;

        const float* xp = x_ + static_cast<std::int64_t>(row) * K + b * 32;
        const float  v  = xp[lane];

        const float amax  = sg_max<float>(sg, sycl::fabs(v));
        const float scale = amax * (1.f / 127.f);
        const float inv   = (amax > 0.f) ? (127.f / amax) : 0.f;
        const int   qi    = static_cast<int>(sycl::rint(v * inv));

        qs_[static_cast<std::int64_t>(row) * K + b * 32 + lane] = static_cast<std::int8_t>(qi);

        const int s_lo = sg_sum<int>(sg, (lane < 16) ? qi : 0);
        const int s_hi = sg_sum<int>(sg, (lane >= 16) ? qi : 0);

        if (lane == 0) {
            d_[static_cast<std::int64_t>(row) * NB + b] = scale;
            const std::int64_t sb = (static_cast<std::int64_t>(row) * NB + b) * 2;
            s_[sb + 0] = s_lo;
            s_[sb + 1] = s_hi;
        }
    }

private:
    std::int8_t*  qs_;
    float*        d_;
    std::int32_t* s_;
    const float*  x_;
    int           total_;
};

template <int K>
void quant_launch(ActQ8View dst, const float* x, int rows, sycl::queue& q) {
    constexpr int NB = K / 32;
    const int total = rows * NB;
    if (total <= 0) return;
    const int groups = div_up(total, kQuantSg);
    q.parallel_for<QuantKernel<K>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups) * kQuantSg * kSyclSg),
                          sycl::range<1>(kQuantSg * kSyclSg)),
        QuantKernel<K>(dst.qs, dst.d, dst.s, x, total));
}

} // namespace

template <int K>
void quant_rows(ActQ8View dst, const float* x, int rows, sycl::queue& q) {
    quant_launch<K>(dst, x, rows, q);
}

template void quant_rows<kD  >(ActQ8View, const float*, int, sycl::queue&);
template void quant_rows<kFFN>(ActQ8View, const float*, int, sycl::queue&);

} // namespace syclk
} // namespace nova
