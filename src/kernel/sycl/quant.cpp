// nova_v2/src/kernel/sycl/quant.cpp — fp32 -> int8（每 32 元素一个 fp32 scale，每 16 一个 int 和）
//
// 这就是 ActQ8 的生产者：qs 每元素 1 B，d 每 32 元素 4 B，s 每 16 元素 4 B。
// 与 CUDA 的 quant_kernel 语义逐位一致（同样的 amax/127 缩放、同样的 __float2int_rn
// -> sycl::rint、同样的"每 16 元素求和"）。
//
// 布局选择：**一个 subgroup 吃一个 32 元素块**，每个 lane 处理 PER = 32/SG 个元素
//（SG=32 → 1 个、SG=16 → 2 个、SG=8 → 4 个，统一走 sycl_common.h 的 q8_block<>）。
// CUDA 版是一个 thread 吃 32 个元素（跨步 128 B 的访问模式）；这里一个 subgroup
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

    [[sycl::reqd_sub_group_size(syclb::kSyclSg)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int NB  = K / 32;
        constexpr int PER = 32 / syclb::kSyclSg;
        const int blk = static_cast<int>(it.get_global_linear_id() / syclb::kSyclSg);
        if (blk >= total_) return;

        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int row  = blk / NB;
        const int b    = blk - row * NB;

        const float* xp = x_ + static_cast<std::int64_t>(row) * K + b * 32;

        // 一个 32 元素块由整个 subgroup 负责；SG<32 时每 lane 持 PER 个元素。
        // 以前这里写死成「1 lane = 1 元素」，SG=16 时每个块只量化了前 16 个、s_hi 恒为 0
        //（sycl_bench §8 的 quant 抽检抓到的：int8 不一致 3327 个）。
        float v[PER];
        #pragma unroll
        for (int t = 0; t < PER; ++t) v[t] = xp[lane + t * syclb::kSyclSg];

        int qi[PER];
        const Q8Block<syclb::kSyclSg> blkq = q8_block<syclb::kSyclSg>(sg, lane, v, qi);

        std::int8_t* qrow = qs_ + static_cast<std::int64_t>(row) * K + b * 32;
        #pragma unroll
        for (int t = 0; t < PER; ++t) {
            qrow[lane + t * syclb::kSyclSg] = static_cast<std::int8_t>(qi[t]);
        }

        if (lane == 0) {
            d_[static_cast<std::int64_t>(row) * NB + b] = blkq.scale;
            const std::int64_t sb = (static_cast<std::int64_t>(row) * NB + b) * 2;
            s_[sb + 0] = blkq.s_lo;
            s_[sb + 1] = blkq.s_hi;
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
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups) * kQuantSg * syclb::kSyclSg),
                          sycl::range<1>(kQuantSg * syclb::kSyclSg)),
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
