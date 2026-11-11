// nova_v2/src/kernel/sycl/q4k_gemv.cpp — Q4_K 权重 × int8 激活的单列 GEMV（decode 主路径）
//
// ============================ 它解决什么问题 ============================
// decode 每个 token 都要把整个权重矩阵读一遍（1.5B 约 954 MiB/token），所以这就是一条
// **带宽受限**的路径：算得再快也没用，能不能把 37 GB/s 喂到 50+ 才是关键。
// 与 CUDA 版 src/kernel/q4k_gemv.cu 逐行同构（同 lane 映射、同累加顺序），便于两边对拍。
//
// ============================ 核心参数 ============================
//   K       每行列数，必须是 256 的倍数（Q4_K 的 super-block 是 256 元素）。
//           取值：kD（1536，hidden 宽）或 kFFN（8960，FFN 宽）。
//   HW      true = 走硬件路径（NOVA_SYCL_DP4A 决定是 dp4a 还是标量），false = 强制标量。
//           注意：**当前 DP4A 恒为关闭**（SYCL 2020 没有 int8×int8→int32 的 dot 形式，
//           见 docs/SYCL_PORT.md §13.9），所以 HW=true 实际等于标量。
//   SPLITS  一行由几个 subgroup 分段扫。取值 1..kGemvRows（默认 1）。
//           1 = 旧映射（A/B 基线）；>1 = 方案 C，已证伪，仅保留作对照臂。
//   STRIPE  段间是否交错（false = 各扫一段连续区）。只在 SPLITS>1 时有意义。
//   PD      预取窗深度，= min(NP, NOVA_SYCL_GEMV_PD)，默认 4。
//           实测 4 最优：8/16 会让 wv[PD] 溢出到私有内存，带宽反降 18%/40%。
//
// ============================ lane / work-group 映射 ============================
//   一个 work-group = kGemvRows 个 subgroup（默认 8 行），
//   一个 subgroup = 一行，lane 与 super-block 内 16 B 的对应关系由 q4k_lane(lane) 给出：
//     lane 的 bit0..2 选 8 个 16 B 子片之一，bit3.. 决定同时抓第几个 super-block。
//   每个 144 B 的 super-block 被 8 个 lane 各读 16 B 的 qs（合并成 128 B 一次传输），
//   而 16 B 的块头（d/dmin/scales）被同一 subgroup 的所有 lane 重复读 —— 同地址广播，
//   硬件合并成 1 个事务，但仍占一个 64 B sector（每块 192 B 流量，这就是 46 GB/s 天花板）。
//
// ============================ 逻辑步骤 ============================
//   1. row = group_id * RWS + (sg_id / SPLITS)；越界的行整个 subgroup 一起跳过（不早退，
//      因为下面有 barrier —— 见"边界检查"）。
//   2. 本段负责的 super-block 区间 [b0, hi)：SPLITS=1 时就是 [0, BPR)。
//   3. 预取窗：一次发 PD 个 16 B 的 qs 载入（wv[t]），再回来算，掩盖 DRAM 延迟。
//   4. 每个块：q4k_lane 取出 4 个 int32 权重 → 与 int8 激活做 dp4a（或标量）→ 乘
//      scale/min → 累加到 acc。
//   5. SPLITS=1：sg_sum 归约后 lane0 写 out[row]；SPLITS>1：先写 local part_，
//      barrier 后由 split==0 的 subgroup 相加写出。
//
// ============================ 边界检查（每一处都在防什么） ============================
//   · `sbi < hi ? sbi : hi - 1`：预取窗会越过段尾，多读的那几次读"本段最后一块"（数据会
//     被丢掉），**故意不读行外**；所以 hi-1 必须 >= 0，由 static_assert(K/kQK >= SPLITS) 保证。
//   · `row < rows_`：行数不是 RWS 的整数倍时，最后一组会有越界行；这些行不写 out，
//     但仍要走到 barrier（所以不能 return）。
//   · `p >= NP || sbi >= hi → continue`：预取窗尾部与段尾的双重裁剪。
//
// ============================ 指针生命周期 ============================
//   w_ / xq_ / xd_ / xs_ / out_ 全是**调用方**（gemv_launch / ChunkedWeight）持有的设备
//   指针，本 kernel 只读不持有：launch 返回后它们仍由 Device/runner 管理，kernel 不释放任何东西。
//   part_ 是 local_accessor（work-group 私有），生命周期 = 本次 launch。
//
// ============================ 硬件差异（CUDA vs SYCL） ============================
//   · CUDA sm_61：warp 32 lane，`__shfl_down_sync` 做归约，`__dp4a` 可用。
//   · Intel Arc：subgroup 由 NOVA_SYCL_SG 决定（默认 16，⚠️ 16 时 kBlocksInFlight=2，
//     与 CUDA 的 4 不同，**A/B 对比时必须同 SG**）；`sycl::reduce_over_group` 做归约。
//   · 头文件里所有 static_assert 都在兜住"SG 必须整除 32"这类跨后端假设。
//
// ============================ 与 llama.cpp 的对应 ============================
//   ggml-sycl/mmvq.cpp   : mul_mat_vec_q<QK_K, QI4_K, block_q4_K, ...>
//   ggml-sycl/vecdotq.hpp: vec_dot_q4_K_q8_1_impl_vmmq()
//   区别只有两点：① 我们的激活自带每 16 元素的 int 和（xs），省掉 dot2 那两步；
//   ② 我们一次发 PD 个 16 B 载入（预取窗），llama.cpp 是逐块串行。

#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

using block_q4_K = nova::block_q4_K;

// 一个 lane 一次抓 16 B 权重；PD 决定同一条 lane 上有几个载入在飞（实际取 min(PD, NP)）。
// 预取窗深度：同一条 lane 上有几个 16 B 载入在飞。4 是原来拍的值，实测 GEMV 只跑到
// 34~38 GB/s（顺序读上限 61.38），怀疑 DRAM 延迟没被掩盖住，所以做成编译期可扫：
//   icpx ... -DNOVA_SYCL_GEMV_PD=8
#ifndef NOVA_SYCL_GEMV_PD
inline constexpr int kPrefetchDepth = 4;
#else
inline constexpr int kPrefetchDepth = NOVA_SYCL_GEMV_PD;
#endif

inline sycl::vec<float, 2> load8f(const float* p) {
    sycl::vec<float, 2> v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

template <int K, bool HW, int SPLITS, bool STRIPE>
class GemvQ4K {
public:
    GemvQ4K(const std::uint8_t* w, const std::int8_t* xq, const float* xd, const std::int32_t* xs,
            float* out, int rows, sycl::local_accessor<float, 1> part)
        : w_(w), xq_(xq), xd_(xd), xs_(xs), out_(out), rows_(rows), part_(part) {}

    [[sycl::reqd_sub_group_size(syclb::kSyclSg)]]
    void operator()(sycl::nd_item<2> it) const {
        constexpr int BPR = K / kQK;                       // 每行 super-block 数
        constexpr int SEG = div_up(BPR, SPLITS);           // 最长的一段（块数）
        constexpr int NP  = div_up(SEG, kBlocksInFlight);  // 每条 lane 的迭代数
        constexpr int PD  = (NP < kPrefetchDepth) ? NP : kPrefetchDepth;
        constexpr int RWS = kGemvRows / SPLITS;            // 一个 work-group 负责几行
        constexpr std::uint32_t kMask = 0x0F0F0F0Fu;

        // 本 WG 的第 sg_id 个 subgroup 负责：第 r_in 行的第 split 段
        const int sg_id = static_cast<int>(it.get_local_id(0));
        const int lane  = static_cast<int>(it.get_local_id(1));
        const int split = sg_id % SPLITS;
        const int r_in  = sg_id / SPLITS;
        const int row   = static_cast<int>(it.get_group(0)) * RWS + r_in;

        const sycl::sub_group sg = it.get_sub_group();
        const Q4KLane L = q4k_lane(lane);

        // 本段的 super-block 区间 [b0, hi)；STRIPE 时区间是整个 [0, BPR)，块按 subgroup 交错分。
        const int b0 = STRIPE ? 0 : q4k_seg_begin(split, SPLITS, BPR);
        const int hi = STRIPE ? BPR : q4k_seg_end(split, SPLITS, BPR);

        float acc = 0.f;

        // 注意: row 越界时不能 return —— 下面有 barrier，必须所有 work-item 一起到。
        if (row < rows_ && hi > b0) {
            const block_q4_K* wrow =
                reinterpret_cast<const block_q4_K*>(w_) + static_cast<std::int64_t>(row) * BPR;

            // PD 深的预取窗：先把 PD 个 16 B 权重发出去，再回来算。
            for (int p0 = 0; p0 < NP; p0 += PD) {
                sycl::vec<std::int32_t, 4> wv[PD];
                #pragma unroll
                for (int t = 0; t < PD; ++t) {
                    const int sbi = block_index(p0 + t, L.block, split);
                    // 越界那几次读的是本段最后一个块（结果会被丢掉），避免读到行外
                    wv[t] = load16((wrow + (sbi < hi ? sbi : hi - 1))->qs + L.woff);
                }
                #pragma unroll
                for (int t = 0; t < PD; ++t) {
                    const int p   = p0 + t;
                    const int sbi = block_index(p, L.block, split);
                    if (p >= NP || sbi >= hi) continue;

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
        }

        if constexpr (SPLITS == 1) {
            // 旧映射：一行一个 subgroup，行内 subgroup 归约就完事
            const float total = sg_sum<float>(sg, acc);
            if (lane == 0 && row < rows_) out_[row] = total;
        } else {
            // 方案 C：每个 subgroup 出部分和 -> local -> 每行第一个 subgroup 的 lane 0 相加写出。
            // 合并顺序固定 split=0,1,2,...，与调度无关，结果可复现。
            const float part = sg_sum<float>(sg, acc);
            if (lane == 0) part_[sg_id] = part;
            it.barrier(sycl::access::fence_space::local_space);
            if (split == 0 && lane == 0 && row < rows_) {
                float total = 0.f;
                #pragma unroll
                for (int s = 0; s < SPLITS; ++s) total += part_[r_in * SPLITS + s];
                out_[row] = total;
            }
        }
    }

private:
    // 第 p 次迭代、第 blk 个 lane 组要读的 super-block（行内绝对编号）
    static constexpr int block_index(int p, int blk, int split) {
        constexpr int BPR = K / kQK;
        if constexpr (STRIPE) {
            // 交错：第 p 轮里 SPLITS 个 subgroup 抓的是相邻的 SPLITS 段，
            // 整个 WG 一轮覆盖 SPLITS*kBlocksInFlight 个连续 super-block
            return (p * SPLITS + split) * kBlocksInFlight + blk;
        } else {
            // 连续分段（方案 C）：第 split 段只管 [b0, b0+SEG) 这一段
            return q4k_seg_begin(split, SPLITS, BPR) + p * kBlocksInFlight + blk;
        }
    }

    const std::uint8_t*  w_;
    const std::int8_t*   xq_;
    const float*         xd_;
    const std::int32_t*  xs_;
    float*               out_;
    int                  rows_;
    sycl::local_accessor<float, 1> part_;
};

template <int K, bool HW, int SPLITS, bool STRIPE>
void gemv_launch_one(const std::uint8_t* w, const ActQ8View& a, float* out, int rows, sycl::queue& q) {
    static_assert(SPLITS >= 1 && SPLITS <= kGemvRows, "GEMV 每行分段数必须 1..GEMV_ROWS");
    static_assert(kGemvRows % SPLITS == 0, "GEMV_ROWS 必须能被每行分段数整除");
    static_assert(K / kQK >= SPLITS, "每行 super-block 数必须 >= 分段数（否则有空段）");

    constexpr int RWS = kGemvRows / SPLITS;      // 一个 WG 负责几行
    const int groups = div_up(rows, RWS);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(static_cast<std::size_t>(kGemvRows)), h);
        h.parallel_for<GemvQ4K<K, HW, SPLITS, STRIPE>>(
            sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(groups) * kGemvRows, syclb::kSyclSg),
                              sycl::range<2>(kGemvRows, syclb::kSyclSg)),
            GemvQ4K<K, HW, SPLITS, STRIPE>(w, a.qs, a.d, a.s, out, rows, part));
    });
}

template <int K, bool HW, int SPLITS, bool STRIPE>
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
        const int rows = static_cast<int>(ch.row1 - ch.row0);
        gemv_launch_one<K, HW, SPLITS, STRIPE>(ch.ptr, a, out + ch.row0, rows, q);
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
    gemv_launch<K, HW, kGemvSplits, kGemvStripe>(w, a, out, q);
}

template <int K, bool HW, int SPLITS, bool STRIPE>
void gemv_variant(const ChunkedWeight& w, const ActQ8View& a, float* out, sycl::queue& q) {
    gemv_launch<K, HW, SPLITS, STRIPE>(w, a, out, q);
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

// A/B 用的显式变体（基线 SPLITS=1 / 2 段 / 方案 C 4 段连续 / 4 段交错），只实例化 HW=true
#define NOVA_SYCL_INST_GEMV_VARIANT(K)                                                        \
    template void gemv_variant<K, true, 1, false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&); \
    template void gemv_variant<K, true, 2, false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&); \
    template void gemv_variant<K, true, 4, false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&); \
    template void gemv_variant<K, true, 4, true >(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);

NOVA_SYCL_INST_GEMV_VARIANT(kD)
NOVA_SYCL_INST_GEMV_VARIANT(kFFN)

} // namespace syclk
} // namespace nova
