// nova_v2/src/kernel/sycl/q4k_gemv4.cpp — 四列批量 GEMV（对应 CUDA 的 gemv4）
//
// 与 q4k_gemv.cpp 的唯一区别：一个 lane 的 16 B 权重解包一次，喂 4 组激活，
// 权重读放大 = 1，dp4a 数 = 4 倍。批量 decode（kMaxBatch=4）走这条。
//
// CUDA 版把低 nibble 重新"拼"成 qlo = (lo&0xF) | ((lo>>8&0xF)<<8) | ...
// 这里直接用 lo & 0x0F0F0F0F —— 两者逐位等价（原式只是把每个字节的低 nibble
// 留在原地），少 12 条指令。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

using block_q4_K = nova::block_q4_K;

inline constexpr int kPrefetchDepth4 = 4;
inline constexpr int kCols4          = 4;

inline sycl::vec<float, 2> load8f(const float* p) {
    sycl::vec<float, 2> v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

template <int K, bool HW>
class Gemv4Q4K {
public:
    Gemv4Q4K(const std::uint8_t* w, const std::int8_t* xq4, const float* xd4, const std::int32_t* xs4,
             float* out, int out_stride, int rows)
        : w_(w), xq4_(xq4), xd4_(xd4), xs4_(xs4), out_(out), out_stride_(out_stride), rows_(rows) {}

    void operator()(sycl::nd_item<2> it) const [[sycl::reqd_sub_group_size(kSyclSg)]] {
        constexpr int BPR = K / kQK;
        constexpr int NP  = div_up(BPR, kBlocksInFlight);
        constexpr std::uint32_t kMask = 0x0F0F0F0Fu;

        const int row = it.get_group(0) * kGemvRows + static_cast<int>(it.get_local_id(0));
        if (row >= rows_) return;

        const int lane = static_cast<int>(it.get_local_id(1));
        const sycl::sub_group sg = it.get_sub_group();
        const Q4KLane L = q4k_lane(lane);

        const block_q4_K* wrow =
            reinterpret_cast<const block_q4_K*>(w_) + static_cast<std::int64_t>(row) * BPR;

        float acc[kCols4] = {0.f, 0.f, 0.f, 0.f};

        for (int p0 = 0; p0 < NP; p0 += kPrefetchDepth4) {
            sycl::vec<std::int32_t, 4> wv[kPrefetchDepth4];
            #pragma unroll
            for (int t = 0; t < kPrefetchDepth4; ++t) {
                const int p = p0 + t;
                int sbi     = p * kBlocksInFlight + L.block;
                if (sbi >= BPR) sbi = BPR - 1;
                wv[t] = load16((wrow + sbi)->qs + L.woff);
            }
            #pragma unroll
            for (int t = 0; t < kPrefetchDepth4; ++t) {
                const int p   = p0 + t;
                const int sbi = p * kBlocksInFlight + L.block;
                if (p >= NP || sbi >= BPR) continue;

                const block_q4_K* b = wrow + sbi;

                std::uint8_t scl, mcl, sch, mch;
                get_scale_min_k4(L.slo,     b->scales, scl, mcl);
                get_scale_min_k4(L.slo + 1, b->scales, sch, mch);
                const float dd = h2f(b->d);
                const float dm = h2f(b->dmin);
                const float dl = dd * static_cast<float>(scl);
                const float ml = dm * static_cast<float>(mcl);
                const float dh = dd * static_cast<float>(sch);
                const float mh = dm * static_cast<float>(mch);

                // 16 B 权重 -> 8 个 dp4a 的入参；4 列复用这 8 个常量
                const std::int32_t lo0 = static_cast<std::int32_t>(static_cast<std::uint32_t>(wv[t].x()) & kMask);
                const std::int32_t lo1 = static_cast<std::int32_t>(static_cast<std::uint32_t>(wv[t].y()) & kMask);
                const std::int32_t lo2 = static_cast<std::int32_t>(static_cast<std::uint32_t>(wv[t].z()) & kMask);
                const std::int32_t lo3 = static_cast<std::int32_t>(static_cast<std::uint32_t>(wv[t].w()) & kMask);
                const std::int32_t hi0 = static_cast<std::int32_t>((static_cast<std::uint32_t>(wv[t].x()) >> 4) & kMask);
                const std::int32_t hi1 = static_cast<std::int32_t>((static_cast<std::uint32_t>(wv[t].y()) >> 4) & kMask);
                const std::int32_t hi2 = static_cast<std::int32_t>((static_cast<std::uint32_t>(wv[t].z()) >> 4) & kMask);
                const std::int32_t hi3 = static_cast<std::int32_t>((static_cast<std::uint32_t>(wv[t].w()) >> 4) & kMask);

                #pragma unroll
                for (int c2 = 0; c2 < kCols4; ++c2) {
                    const std::int8_t* ab = xq4_ + static_cast<std::int64_t>(c2) * K + static_cast<std::int64_t>(sbi) * kQK;
                    const sycl::vec<std::int32_t, 4> al = load16(ab + L.alo);
                    const sycl::vec<std::int32_t, 4> ah = load16(ab + L.ahi);

                    const int sm_lo = xs4_[static_cast<std::int64_t>(c2) * (K / 16) + sbi * 16 + (L.alo >> 4)];
                    const int sm_hi = xs4_[static_cast<std::int64_t>(c2) * (K / 16) + sbi * 16 + (L.ahi >> 4)];
                    const sycl::vec<float, 2> dx =
                        load8f(xd4_ + static_cast<std::int64_t>(c2) * (K / 32) + sbi * 8 + L.slo);

                    int dlo = 0, dhi = 0;
                    dlo = d4<HW>(lo0, al.x(), dlo);
                    dlo = d4<HW>(lo1, al.y(), dlo);
                    dlo = d4<HW>(lo2, al.z(), dlo);
                    dlo = d4<HW>(lo3, al.w(), dlo);
                    dhi = d4<HW>(hi0, ah.x(), dhi);
                    dhi = d4<HW>(hi1, ah.y(), dhi);
                    dhi = d4<HW>(hi2, ah.z(), dhi);
                    dhi = d4<HW>(hi3, ah.w(), dhi);

                    acc[c2] += dx[0] * (dl * static_cast<float>(dlo) - ml * static_cast<float>(sm_lo));
                    acc[c2] += dx[1] * (dh * static_cast<float>(dhi) - mh * static_cast<float>(sm_hi));
                }
            }
        }

        #pragma unroll
        for (int c2 = 0; c2 < kCols4; ++c2) {
            const float total = sg_sum<float>(sg, acc[c2]);
            if (lane == 0) out_[static_cast<std::int64_t>(c2) * out_stride_ + row] = total;
        }
    }

private:
    const std::uint8_t* w_;
    const std::int8_t*  xq4_;
    const float*        xd4_;
    const std::int32_t* xs4_;
    float*              out_;
    int                 out_stride_;
    int                 rows_;
};

template <int K, bool HW>
void gemv4_launch(const ChunkedWeight& w, const std::int8_t* xq4, const float* xd4, const std::int32_t* xs4,
                  float* out4, sycl::queue& q) {
    const std::int64_t row_bytes = static_cast<std::int64_t>(K / kQK) * kBlockBytes;
    if (w.row_bytes() != row_bytes) {
        char m[192];
        std::snprintf(m, sizeof(m), "gemv4<K=%d>: 权重行字节 %lld != %lld",
                      K, static_cast<long long>(w.row_bytes()), static_cast<long long>(row_bytes));
        NOVA_ABORT(m);
    }
    const std::int64_t stride = w.rows();
    for (int c = 0; c < w.n_chunks(); ++c) {
        const ChunkedWeight::Chunk& ch = w.chunk(c);
        const int rows   = static_cast<int>(ch.row1 - ch.row0);
        const int groups = div_up(rows, kGemvRows);
        q.parallel_for<Gemv4Q4K<K, HW>>(
            sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(groups) * kGemvRows, kSyclSg),
                              sycl::range<2>(kGemvRows, kSyclSg)),
            Gemv4Q4K<K, HW>(ch.ptr, xq4, xd4, xs4, out4 + ch.row0,
                            static_cast<int>(stride), rows));
    }
}

} // namespace

template <int K, bool HW>
void gemv4(const ChunkedWeight& w, const std::int8_t* xq4, const float* xd4, const std::int32_t* xs4,
           float* out4, sycl::queue& q) {
    gemv4_launch<K, HW>(w, xq4, xd4, xs4, out4, q);
}

#define NOVA_SYCL_INST_GEMV4(K)                                                                        \
    template void gemv4<K, true >(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&); \
    template void gemv4<K, false>(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&);

NOVA_SYCL_INST_GEMV4(kD)
NOVA_SYCL_INST_GEMV4(kFFN)

} // namespace syclk
} // namespace nova
