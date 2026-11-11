// nova_v2/src/kernel/sycl/q4k_gemv_soa.cpp — SoA 布局的 Q4_K GEMV（decode 单列）
//
// 为什么单独一个内核：旧内核带着 SPLITS/STRIPE/PRECOMP 三条实验分支，改它风险高；
// SoA 的收益只取决于"怎么读"，所以这里做一条干净路径，数值与原内核逐位等价。
//
// 布局（加载期由 repack 生成，常量在 kernels.h）：
//   qs  : [rows][BPR][128 B]                每个 super-block 的量化权重
//   hdr : [rows][NP][kInFlight][16 B]        块头，**按内核访问顺序**排
//        所以一条 lane 在一次预取窗（PD=4）里要用的 4 个块头是物理相邻的 64 B
//        = 恰好 1 个 64 B sector。按块号顺序排的话它们会隔开 kInFlight 个 16 B，
//        跨 2 个 sector，收益腰斩 —— 这是实测过 sector 模型之后的选择。
//
// 每 144 B 块的流量：qs 128 B（2 sectors，8 lane 完美合并）+ 块头 16/4 B（4 块共 1 sector）
//   = 2.25 sectors = 144 B   ← 不再是 AoS 的 3 sectors = 192 B
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

// 16 B 块头：d、dmin、6-bit scales/min（与 block_q4_K 的前 16 B 逐字节相同）
struct Q4KHdr {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t  scales[12];
};
static_assert(sizeof(Q4KHdr) == 16, "Q4KHdr 必须是 16 B");

#ifndef NOVA_SYCL_GEMV_PD
inline constexpr int kPrefetchDepthSoA = 4;     // PD 扫描证明 4 最优（8/16 反而更慢）
#else
inline constexpr int kPrefetchDepthSoA = NOVA_SYCL_GEMV_PD;
#endif

inline sycl::vec<float, 2> load8f_soa(const float* p) {
    sycl::vec<float, 2> v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

template <int K, bool HW>
class GemvSoA {
public:
    GemvSoA(const std::uint8_t* qs, const std::uint8_t* hdr, const std::int8_t* xq,
            const float* xd, const std::int32_t* xs, float* out, int rows)
        : qs_(qs), hdr_(hdr), xq_(xq), xd_(xd), xs_(xs), out_(out), rows_(rows) {}

    [[sycl::reqd_sub_group_size(syclb::kSyclSg)]]
    void operator()(sycl::nd_item<2> it) const {
        constexpr int BPR = K / kQK;
        constexpr int NP  = soa_np(BPR);
        constexpr int PD  = (NP < kPrefetchDepthSoA) ? NP : kPrefetchDepthSoA;
        constexpr std::uint32_t kMask = 0x0F0F0F0Fu;

        const int r_in = static_cast<int>(it.get_local_id(0));
        const int lane = static_cast<int>(it.get_local_id(1));
        const int row  = static_cast<int>(it.get_group(0)) * kGemvRows + r_in;
        if (row >= rows_) return;

        const sycl::sub_group sg = it.get_sub_group();
        const Q4KLane L = q4k_lane(lane);

        const std::uint8_t* qs_row = qs_ + static_cast<std::int64_t>(row) * soa_qs_row_bytes(BPR);
        const Q4KHdr* hdr_row = reinterpret_cast<const Q4KHdr*>(
            hdr_ + static_cast<std::int64_t>(row) * soa_hdr_row_bytes(BPR));

        float acc = 0.f;
        for (int p0 = 0; p0 < NP; p0 += PD) {
            // 关键：本 lane 组这 PD 次迭代的块头在 hdr 里连续 → 一次 64 B 载入（1 sector）
            Q4KHdr hg[PD];
            #pragma unroll
            for (int t = 0; t < PD; ++t) {
                const int p   = (p0 + t < NP) ? (p0 + t) : (NP - 1);
                const int idx = L.block * NP + p;   // 每条 lane 的块头在 hdr 里连续（4 个 = 64 B）
                hg[t] = hdr_row[idx];
            }

            // qs：PD 个 16 B 载入先发出去（与原内核相同的预取窗）
            sycl::vec<std::int32_t, 4> wv[PD];
            #pragma unroll
            for (int t = 0; t < PD; ++t) {
                const int p = (p0 + t < NP) ? (p0 + t) : (NP - 1);
                const int b = p * kSoAInFlight + L.block;
                wv[t] = load16(qs_row + static_cast<std::int64_t>(b) * 128 + L.woff);
            }

            #pragma unroll
            for (int t = 0; t < PD; ++t) {
                const int p = p0 + t;
                if (p >= NP) continue;
                const int b = p * kSoAInFlight + L.block;
                if (b >= BPR) continue;

                const std::int8_t* ab = xq_ + static_cast<std::int64_t>(b) * kQK;
                const sycl::vec<std::int32_t, 4> al = load16(ab + L.alo);
                const sycl::vec<std::int32_t, 4> ah = load16(ab + L.ahi);

                const int sm_lo = xs_[b * 16 + (L.alo >> 4)];
                const int sm_hi = xs_[b * 16 + (L.ahi >> 4)];
                const sycl::vec<float, 2> dx = load8f_soa(xd_ + b * 8 + L.slo);

                std::uint8_t scl, mcl, sch, mch;
                get_scale_min_k4(L.slo,     hg[t].scales, scl, mcl);
                get_scale_min_k4(L.slo + 1, hg[t].scales, sch, mch);
                const float dd = h2f(hg[t].d);
                const float dm = h2f(hg[t].dmin);

                const std::int32_t w0 = wv[t].x(), w1 = wv[t].y(), w2 = wv[t].z(), w3 = wv[t].w();
                int dlo = 0, dhi = 0;
                dlo = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w0) & kMask), al.x(), dlo);
                dlo = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w1) & kMask), al.y(), dlo);
                dlo = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w2) & kMask), al.z(), dlo);
                dlo = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w3) & kMask), al.w(), dlo);
                dhi = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w0) >> 4) & kMask), ah.x(), dhi);
                dhi = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w1) >> 4) & kMask), ah.y(), dhi);
                dhi = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w2) >> 4) & kMask), ah.z(), dhi);
                dhi = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w3) >> 4) & kMask), ah.w(), dhi);

                acc += dx[0] * ((dd * static_cast<float>(scl)) * static_cast<float>(dlo) -
                                (dm * static_cast<float>(mcl)) * static_cast<float>(sm_lo));
                acc += dx[1] * ((dd * static_cast<float>(sch)) * static_cast<float>(dhi) -
                                (dm * static_cast<float>(mch)) * static_cast<float>(sm_hi));
            }
        }

        const float total = sg_sum<float>(sg, acc);
        if (lane == 0) out_[row] = total;
    }

private:
    const std::uint8_t* qs_;
    const std::uint8_t* hdr_;
    const std::int8_t*  xq_;
    const float*        xd_;
    const std::int32_t* xs_;
    float*              out_;
    int                 rows_;
};

} // namespace

template <int K, bool HW>
sycl::event gemv_soa(const SoAWeightView& w, const ActQ8View& a, float* out, sycl::queue& q) {
    const int rows = static_cast<int>(w.rows);
    if (rows <= 0) return sycl::event{};
    const int groups = div_up(rows, kGemvRows);
    // 不把 submit 的返回值往外递：CPU 垫片的 submit 返回 void，而生产路径的计时走
    // Scope 的队列边界事件，用不到这个 event。
    q.submit([&](sycl::handler& h) {
        h.parallel_for<GemvSoA<K, HW>>(
            sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(groups) * kGemvRows, syclb::kSyclSg),
                              sycl::range<2>(kGemvRows, syclb::kSyclSg)),
            GemvSoA<K, HW>(w.qs, w.hdr, a.qs, a.d, a.s, out, rows));
    });
    return sycl::event{};
}

// 显式实例化：hidden 宽（q/k/v/o/gate/up 用 K=kD）与 ffn 宽（down 用 K=kFFN）
template sycl::event gemv_soa<kD, true>(const SoAWeightView&, const ActQ8View&, float*, sycl::queue&);
template sycl::event gemv_soa<kFFN, true>(const SoAWeightView&, const ActQ8View&, float*, sycl::queue&);

} // namespace syclk
} // namespace nova
