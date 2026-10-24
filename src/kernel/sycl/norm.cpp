// nova_v2/src/kernel/sycl/norm.cpp — RMSNorm（含融合量化版）
//
// 结构对照 ggml-sycl/norm.cpp : rms_norm_f32() —— 通用实现是"1 个 subgroup 一行 +
// 可选 local 归约"。我们的 D 是编译期常量，所以：
//   * 每行一个 work-group（kRmsWg=256），两级归约：subgroup 内 reduce_over_group
//     -> local_accessor 汇总 kNSG 个部分和 -> 每个线程各自把 8 个部分和加起来。
//     （不设第二个 barrier：8 个 float 的广播读比再归约一次便宜。）
//   * 融合版 rmsnorm_quant 把 quant 塞进同一个 kernel，decode 每层省一次 dispatch。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

inline constexpr int kNSG = kRmsWg / kSyclSg;

} // namespace

// kernel 名（parallel_for<Name> 需要一个完整类型；这里只是标签）
struct RmsNormKernel {};
struct RmsNormQuantKernel {};

void rms_norm(float* y, const float* x, const float* w, int rows, sycl::queue& q) {
    constexpr int K = kD;
    if (rows <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sred(sycl::range<1>(kNSG), h);
        h.parallel_for<RmsNormKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(rows) * kRmsWg),
                              sycl::range<1>(kRmsWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSyclSg)]] {
                const int row = static_cast<int>(it.get_group(0));
                const int tid = static_cast<int>(it.get_local_id(0));
                const sycl::sub_group sg = it.get_sub_group();
                const float* xr = x + static_cast<std::int64_t>(row) * K;
                float*       yr = y + static_cast<std::int64_t>(row) * K;

                float ss = 0.f;
                for (int i = tid; i < K; i += kRmsWg) ss = sycl::fma(xr[i], xr[i], ss);
                const float part = sg_sum<float>(sg, ss);
                if (tid % kSyclSg == 0) sred[tid / kSyclSg] = part;
                it.barrier(sycl::access::fence_space::local_space);

                float total = 0.f;
                #pragma unroll
                for (int i = 0; i < kNSG; ++i) total += sred[i];
                const float scale = sycl::rsqrt(total * (1.f / static_cast<float>(K)) + kEps);

                for (int i = tid; i < K; i += kRmsWg) yr[i] = xr[i] * scale * w[i];
            });
    });
}

void rmsnorm_quant(float* y, ActQ8View dst, const float* x, const float* w, int rows, sycl::queue& q) {
    constexpr int K  = kD;
    constexpr int NB = K / 32;
    if (rows <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sred(sycl::range<1>(kNSG), h);
        h.parallel_for<RmsNormQuantKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(rows) * kRmsWg),
                              sycl::range<1>(kRmsWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSyclSg)]] {
                const int row  = static_cast<int>(it.get_group(0));
                const int tid  = static_cast<int>(it.get_local_id(0));
                const sycl::sub_group sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = tid / kSyclSg;

                const float* xr = x + static_cast<std::int64_t>(row) * K;
                float*       yr = y + static_cast<std::int64_t>(row) * K;

                float ss = 0.f;
                for (int i = tid; i < K; i += kRmsWg) ss = sycl::fma(xr[i], xr[i], ss);
                const float part = sg_sum<float>(sg, ss);
                if (lane == 0) sred[sgid] = part;
                it.barrier(sycl::access::fence_space::local_space);

                float total = 0.f;
                #pragma unroll
                for (int i = 0; i < kNSG; ++i) total += sred[i];
                const float scale = sycl::rsqrt(total * (1.f / static_cast<float>(K)) + kEps);

                std::int8_t* qrow = dst.qs + static_cast<std::int64_t>(row) * K;
                for (int b = sgid; b < NB; b += kNSG) {
                    const int   i = b * 32 + lane;
                    const float v = xr[i] * scale * w[i];
                    yr[i] = v;

                    const float amax = sg_max<float>(sg, sycl::fabs(v));
                    const float sc   = amax * (1.f / 127.f);
                    const float inv  = (amax > 0.f) ? (127.f / amax) : 0.f;
                    const int   qi   = static_cast<int>(sycl::rint(v * inv));
                    qrow[i] = static_cast<std::int8_t>(qi);

                    const int s_lo = sg_sum<int>(sg, (lane < 16) ? qi : 0);
                    const int s_hi = sg_sum<int>(sg, (lane >= 16) ? qi : 0);
                    if (lane == 0) {
                        dst.d[static_cast<std::int64_t>(row) * NB + b] = sc;
                        const std::int64_t sb = (static_cast<std::int64_t>(row) * NB + b) * 2;
                        dst.s[sb + 0] = s_lo;
                        dst.s[sb + 1] = s_hi;
                    }
                }
            });
    });
}

} // namespace syclk
} // namespace nova
