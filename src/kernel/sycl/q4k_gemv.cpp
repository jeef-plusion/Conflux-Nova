// nova_v2/src/kernel/sycl/q4k_gemv.cpp — Q4_K x int8 的 GEMV（decode 主路径）
//
// 结构直接对照 src/kernel/q4k_gemv.cu（CUDA sm_61 版），映射关系：
//   1 个 subgroup(32 lane) = 1 行；lane 固定持 16 B 权重（= 32 个 4-bit 权重）；
//   lane 的 8 个子片（c = lane & 7）覆盖一个 256-super-block 的 qs[128]；
//   4 个 lane 组（lane >> 3）同时抓 4 个 super-block（kBlocksInFlight）。
//
// 与 llama.cpp 的对应关系（详见 docs/SYCL_PORT.md）：
//   ggml-sycl/mmvq.cpp : mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
//                                        VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
//   ggml-sycl/vecdotq.hpp : vec_dot_q4_K_q8_1_impl_vmmq()
// 区别只有两点：① 我们的激活自带每 16 元素的 int 和（xs），省掉 dot2 = dp4a(0x01010101,u)
// 那两步；② 我们一次发 4 个 16 B 载入（PD=4 的预取窗），llama.cpp 是逐块串行。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

using block_q4_K = nova::block_q4_K;

// 一个 lane 一次抓 16 B 权重；PD 决定同一条 lane 上有几个载入在飞。
inline constexpr int kPrefetchDepth = 4;

inline sycl::vec<float, 2> load8f(const float* p) {
    sycl::vec<float, 2> v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

template <int K, bool HW>
class GemvQ4K {
public:
    GemvQ4K(const std::uint8_t* w, const std::int8_t* xq, const float* xd, const std::int32_t* xs,
            float* out, int rows)
        : w_(w), xq_(xq), xd_(xd), xs_(xs), out_(out), rows_(rows) {}

    void operator()(sycl::nd_item<2> it) const [[sycl::reqd_sub_group_size(kSyclSg)]] {
        constexpr int BPR = K / kQK;                                  // 每行 super-block 数
        constexpr int NP  = div_up(BPR, kBlocksInFlight);             // 每条 lane 的迭代数
        constexpr std::uint32_t kMask = 0x0F0F0F0Fu;

        const int row = it.get_group(0) * kGemvRows + static_cast<int>(it.get_local_id(0));
        if (row >= rows_) return;

        const int lane = static_cast<int>(it.get_local_id(1));
        const sycl::sub_group sg = it.get_sub_group();
        const Q4KLane L = q4k_lane(lane);

        const block_q4_K* wrow =
            reinterpret_cast<const block_q4_K*>(w_) + static_cast<std::int64_t>(row) * BPR;

        float acc = 0.f;

        // PD 深的预取窗：先把 4 个 16 B 权重发出去，再回来算。
        for (int p0 = 0; p0 < NP; p0 += kPrefetchDepth) {
            sycl::vec<std::int32_t, 4> wv[kPrefetchDepth];
            #pragma unroll
            for (int t = 0; t < kPrefetchDepth; ++t) {
                const int p   = p0 + t;
                int sbi       = p * kBlocksInFlight + L.block;
                if (sbi >= BPR) sbi = BPR - 1;      // 越界那几次的结果会被丢掉
                wv[t] = load16((wrow + sbi)->qs + L.woff);
            }
            #pragma unroll
            for (int t = 0; t < kPrefetchDepth; ++t) {
                const int p   = p0 + t;
                const int sbi = p * kBlocksInFlight + L.block;
                if (p >= NP || sbi >= BPR) continue;

                const block_q4_K* b = wrow + sbi;
                const std::int8_t* ab = xq_ + static_cast<std::int64_t>(sbi) * kQK;
                const sycl::vec<std::int32_t, 4> al = load16(ab + L.alo);
                const sycl::vec<std::int32_t, 4> ah = load16(ab + L.ahi);

                const int    sm_lo = xs_[sbi * 16 + (L.alo >> 4)];
                const int    sm_hi = xs_[sbi * 16 + (L.ahi >> 4)];
                const sycl::vec<float, 2> dx = load8f(xd_ + sbi * 8 + L.slo);

                std::uint8_t scl, mcl, sch, mch;
                get_scale_min_k4(L.slo,     b->scales, scl, mcl);
                get_scale_min_k4(L.slo + 1, b->scales, sch, mch);
                const float dd = h2f(b->d);
                const float dm = h2f(b->dmin);

                const std::int32_t w0 = wv[t].x();
                const std::int32_t w1 = wv[t].y();
                const std::int32_t w2 = wv[t].z();
                const std::int32_t w3 = wv[t].w();

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
    const std::uint8_t*  w_;
    const std::int8_t*   xq_;
    const float*         xd_;
    const std::int32_t*  xs_;
    float*               out_;
    int                  rows_;
};

template <int K, bool HW>
void gemv_launch(const ChunkedWeight& w, const ActQ8View& a, float* out, sycl::queue& q) {
    const std::int64_t row_bytes = static_cast<std::int64_t>(K / kQK) * kBlockBytes;
    if (w.row_bytes() != row_bytes) {
        char m[192];
        std::snprintf(m, sizeof(m), "gemv<K=%d>: 权重行字节 %lld != K/256*144=%lld",
                      K, static_cast<long long>(w.row_bytes()), static_cast<long long>(row_bytes));
        NOVA_ABORT(m);
    }
    for (int c = 0; c < w.n_chunks(); ++c) {
        const ChunkedWeight::Chunk& ch = w.chunk(c);
        const int rows   = static_cast<int>(ch.row1 - ch.row0);
        const int groups = div_up(rows, kGemvRows);
        q.parallel_for<GemvQ4K<K, HW>>(
            sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(groups) * kGemvRows, kSyclSg),
                              sycl::range<2>(kGemvRows, kSyclSg)),
            GemvQ4K<K, HW>(ch.ptr, a.qs, a.d, a.s, out + ch.row0, rows));
    }
}

// ---------------------------------------------------------------------------
// dp4a 微基准（对应 docs/BACKEND_INTEL_255H.md 的 E1）
//   纯寄存器循环，工作集 0 字节：8 条 dp4a / 迭代，标量版是同样的 32 个乘加。
//   这个数字只回答一个问题：dp4a 是不是真的落在一条指令上。
//   它与 GEMV 的端到端加速比是两个不同的量（后者被权重带宽压住）。
// ---------------------------------------------------------------------------
template <bool HW>
class Dp4aMicro {
public:
    Dp4aMicro(int iters, std::int32_t* out, int count) : iters_(iters), out_(out), count_(count) {}

    void operator()(sycl::nd_item<1> it) const {
        const int gid = static_cast<int>(it.get_global_linear_id());
        if (gid >= count_) return;      // 发射范围按 256 向上取整，多出来的 work-item 直接退出
        std::int32_t w = static_cast<std::int32_t>(static_cast<std::uint32_t>(gid) * 2654435761u) | 1;
        std::int32_t acc = 0;
        constexpr std::uint32_t kM = 0x0F0F0F0Fu;
        for (int i = 0; i < iters_; ++i) {
            acc = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w) & kM), w, acc);
            acc = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w) >> 4) & kM), w ^ 0x5A5A5A5A, acc);
            acc = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w + 1) & kM), w + 1, acc);
            acc = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w + 1) >> 4) & kM), w ^ 0x0F0F0F0F, acc);
            acc = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w + 2) & kM), w + 2, acc);
            acc = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w + 2) >> 4) & kM), w ^ 0x33333333, acc);
            acc = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w + 3) & kM), w + 3, acc);
            acc = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w + 3) >> 4) & kM), w ^ 0x00FF00FF, acc);
            w += 0x01010101;
        }
        out_[gid] = acc;
    }

private:
    int           iters_;
    std::int32_t* out_;
    int           count_;
};

} // namespace

template <int K, bool HW>
void gemv(const ChunkedWeight& w, const ActQ8View& a, float* out, sycl::queue& q) {
    gemv_launch<K, HW>(w, a, out, q);
}

void dp4a_micro(bool hw, int iters, int work_items, std::int32_t* out, sycl::queue& q) {
    if (work_items <= 0) return;
    const int n = ((work_items + 255) / 256) * 256;   // nd_range 要求 global 是 local 的整数倍
    if (hw) {
        q.parallel_for<Dp4aMicro<true>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(n)), sycl::range<1>(256)),
            Dp4aMicro<true>(iters, out, work_items));
    } else {
        q.parallel_for<Dp4aMicro<false>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(n)), sycl::range<1>(256)),
            Dp4aMicro<false>(iters, out, work_items));
    }
}

#define NOVA_SYCL_INST_GEMV(K)                                                       \
    template void gemv<K, true >(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&); \
    template void gemv<K, false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);

NOVA_SYCL_INST_GEMV(kD)
NOVA_SYCL_INST_GEMV(kFFN)

} // namespace syclk
} // namespace nova
