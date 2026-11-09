// nova_v2/src/kernel/sycl/sycl_common.h — SYCL 算子层公共件
//
// 这里放四类东西，全部 header-only：
//   1. 模型形状（复用 nova/model/matrix.h 的唯一真源，与 CUDA 版同一个 ModelSpec）；
//   2. Q4_K 的 lane 映射（与 src/kernel/q4k_gemv.cu 逐行同构）+ fp16 解包 + dp4a 分派；
//   3. subgroup 归约 / 16 B 向量装载小件；
//   4. 调参旋钮（全部 #ifndef，build_sycl.bat 可用 -D 覆盖）。
//
// 约定：kernel 里不出现动态分配、不出现异常、不出现 printf。
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstddef>

#include "nova/config.h"
#include "nova/model/matrix.h"     // ModelSpec / Qwen25_7B / Qwen25_1p5B ...
#include "nova/quant/q4_k.h"       // block_q4_K / get_scale_min_k4 / kQK
#include "device.h"                // profile / syclb::kSyclSg / NOVA_SYCL_DP4A

// ============================================================================
// 0. 本二进制焊死哪个模型（与 CUDA 的 -DNOVA_MODEL_* 同一个意思）
// ============================================================================
#if defined(NOVA_SYCL_MODEL_QWEN25_1P5B)
using SyclActiveModel = nova::Qwen25_1p5B;
inline constexpr const char* kSyclModelName = "Qwen2.5-1.5B-Instruct";
#elif defined(NOVA_SYCL_MODEL_R1_LLAMA8B)
using SyclActiveModel = nova::R1_Llama8B;
inline constexpr const char* kSyclModelName = "DeepSeek-R1-Distill-Llama-8B";
#elif defined(NOVA_SYCL_MODEL_R1_QWEN7B)
using SyclActiveModel = nova::R1_Qwen7B;
inline constexpr const char* kSyclModelName = "DeepSeek-R1-Distill-Qwen-7B";
#else
using SyclActiveModel = nova::Qwen25_7B;
inline constexpr const char* kSyclModelName = "Qwen2.5-7B-Instruct";
#endif

namespace nova {
namespace syclk {

using Model = SyclActiveModel;

inline constexpr int kD    = Model::D;
inline constexpr int kNH   = Model::NH;
inline constexpr int kNKV  = Model::NKV;
inline constexpr int kHD   = Model::HD;
inline constexpr int kFFN  = Model::FFN;
inline constexpr int kV    = Model::V;
inline constexpr int kL    = Model::L;
inline constexpr int kQKV  = kNH * kHD;
inline constexpr int kKV   = kNKV * kHD;
inline constexpr float kEps = Model::eps;
inline constexpr int kTheta = Model::theta;

static_assert(kD % kQK == 0, "hidden 必须是 256 的倍数（Q4_K 无尾块）");
static_assert(kFFN % kQK == 0, "ffn 必须是 256 的倍数");
static_assert((kQKV % 8) == 0 && (kKV % 8) == 0 && (kD % 8) == 0, "行数必须 8 对齐（无尾守卫）");
static_assert(kHD == 64 || kHD == 128, "head_dim 只支持 64 / 128");

// ============================================================================
// 1. 调参旋钮（默认值 = docs/SYCL_PORT.md §5 的推荐值；全部可在命令行覆盖）
// ============================================================================
#ifndef NOVA_SYCL_GEMV_ROWS       // 每个 work-group 里有几个 subgroup（= 8 / SPLIT 行）
#  define NOVA_SYCL_GEMV_ROWS 8
#endif
#ifndef NOVA_SYCL_GEMV_SPLIT      // 每行由几个 subgroup 分段扫（1 = 旧映射）
// 默认 1：方案 C（=4）在目标机 Arc 130T 上实测**显著更慢**，已回退。
// 2026-11-09 实测（1.5B, SG=16, 2000 次发射 × 3 轮，同进程交替，σ<1%）：
//   旧映射 36.07 us = 36.79 GB/s | 4 段连续 37.06 us (-2.7%) | 2 段 41.99 us (-14.1%) | 4 段交错 38.99 us (-7.5%)
// 见 docs/PERF_BASELINE.md §8。要复现实验就 -DNOVA_SYCL_GEMV_SPLIT=4。
#  define NOVA_SYCL_GEMV_SPLIT 1
#endif
#ifndef NOVA_SYCL_GEMV_STRIPE     // 1 = 4 个 subgroup 交错扫一行（段间不连续，实验臂）
#  define NOVA_SYCL_GEMV_STRIPE 0
#endif
#ifndef NOVA_SYCL_RMS_WG          // RMSNorm 每行的 work-group 大小
#  define NOVA_SYCL_RMS_WG 256
#endif
#ifndef NOVA_SYCL_QUANT_SG        // 量化：每个 WG 里几个 subgroup（各自吃一个 32 元素块）
#  define NOVA_SYCL_QUANT_SG 8
#endif
#ifndef NOVA_SYCL_ATTN_DEC_WG     // decode split-K：每个 WG 的 work-item 数（4 个 subgroup）
#  define NOVA_SYCL_ATTN_DEC_WG 128
#endif
#ifndef NOVA_SYCL_ATTN_SPLIT      // decode 的最大 split 数
#  define NOVA_SYCL_ATTN_SPLIT 32
#endif
#ifndef NOVA_SYCL_PREFILL_TQ      // prefill：一个 WG 覆盖几个 query（= subgroup 数）
#  define NOVA_SYCL_PREFILL_TQ 8
#endif
#ifndef NOVA_SYCL_PREFILL_KC      // prefill：每次从 KV 里 stage 多少个 key
#  define NOVA_SYCL_PREFILL_KC 16
#endif
#ifndef NOVA_SYCL_ROPE_NEOX       // 1 = NeoX 布局（Qwen2），0 = 交错布局（Llama）
#  define NOVA_SYCL_ROPE_NEOX 1
#endif

inline constexpr int kGemvRows   = NOVA_SYCL_GEMV_ROWS;
inline constexpr int kGemvSplits = NOVA_SYCL_GEMV_SPLIT;                       // 每行几个 subgroup（方案 C = 4）
inline constexpr bool kGemvStripe = (NOVA_SYCL_GEMV_STRIPE != 0);              // 段间取"交错"还是"连续"
inline constexpr int kGemvRowsPerWg = kGemvRows / kGemvSplits;                 // 一个 WG 负责几行（8/4 = 2）
inline constexpr int kRmsWg      = NOVA_SYCL_RMS_WG;
inline constexpr int kQuantSg    = NOVA_SYCL_QUANT_SG;
inline constexpr int kAttnDecWg  = NOVA_SYCL_ATTN_DEC_WG;
inline constexpr int kAttnSplit  = NOVA_SYCL_ATTN_SPLIT;
inline constexpr int kPrefillTQ  = NOVA_SYCL_PREFILL_TQ;
inline constexpr int kPrefillKC  = NOVA_SYCL_PREFILL_KC;
inline constexpr bool kRopeNeox  = (NOVA_SYCL_ROPE_NEOX != 0);

static_assert(kGemvRows > 0 && kGemvRows * syclb::kSyclSg <= 1024, "GEMV work-group 超 1024");
static_assert(kGemvRows * syclb::kSyclSg <= 512, "GEMV work-group 太大会压 occupancy");
static_assert(kGemvSplits >= 1 && kGemvSplits <= kGemvRows, "GEMV 每行分段数必须 1..GEMV_ROWS");
static_assert((kGemvSplits & (kGemvSplits - 1)) == 0, "GEMV 每行分段数必须是 2 的幂（1/2/4/8）");
static_assert(kGemvRows % kGemvSplits == 0, "GEMV_ROWS 必须能被每行分段数整除");
static_assert(kRmsWg % syclb::kSyclSg == 0 && kRmsWg <= 1024, "RMSNorm WG 必须是 subgroup 的整数倍");
static_assert(kAttnDecWg % syclb::kSyclSgAttn == 0, "decode split-K 的 WG 必须是 subgroup 的整数倍");
static_assert(kAttnDecWg / syclb::kSyclSgAttn >= 1 && kAttnDecWg / syclb::kSyclSgAttn <= 8, "decode 每 WG 的 subgroup 数 1..8");
static_assert(kPrefillTQ * syclb::kSyclSgAttn <= 1024, "prefill WG 超 1024");
// quant / rmsnorm_quant / swiglu_quant 都是「一个 subgroup 量化一个 32 元素块」，
// 每 lane 处理 32/SG 个元素；SG 必须能整除 32（8/16/32 都行，其它值直接编译期挡）。
static_assert(32 % syclb::kSyclSg == 0, "subgroup 必须能整除 Q4_K 的 32 元素子块");
// prefill 的 KV tile 必须塞进 32 KB local
static_assert(2 * kPrefillKC * kHD * 4 <= nova::syclb::profile::kLocalMemBytes,
              "prefill KV tile 超过 32 KiB local 上限");

// ============================================================================
// 2. 小工具
// ============================================================================
inline constexpr int div_up(int a, int b) { return (a + b - 1) / b; }

// fp16 -> fp32，纯整数位运算，device/host 同一份实现。
// 不用 sycl::half 是为了不依赖 aspect::fp16 的算术支持；不用 std::ldexp 是因为它在 device 端不可靠。
inline float h2f(std::uint16_t h) {
    const std::uint32_t u    = static_cast<std::uint32_t>(h);
    const std::uint32_t sign = (u & 0x8000u) << 16;
    const std::uint32_t exp  = (u >> 10) & 0x1Fu;
    const std::uint32_t man  = u & 0x3FFu;
    std::uint32_t bits;
    if (exp != 0u) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);      // 正规数（含 inf/nan）
    } else if (man == 0u) {
        bits = sign;                                            // ±0
    } else {
        std::uint32_t e = 0u, m = man;                          // 次正规数：规范化
        while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
        bits = sign | ((113u - e) << 23) | ((m & 0x3FFu) << 13);
    }
    union { std::uint32_t u; float f; } cv;
    cv.u = bits;
    return cv.f;
}

// 4 个打包 int8 -> vec<int8_t,4>
inline sycl::vec<std::int8_t, 4> as_i8x4(std::int32_t w) {
    sycl::vec<std::int32_t, 1> v(w);
    return v.as<sycl::vec<std::int8_t, 4>>();
}

// ---- dp4a --------------------------------------------------------------
// 标量路径：与 CUDA __dp4a / dpct::dp4a 逐位同结果（4 个 int8 乘加）。
inline int d4_scalar(std::int32_t w, std::int32_t a, int c) {
    int r = c;
    r += (w & 0xF) * static_cast<int>(static_cast<std::int8_t>(a & 0xFF));
    r += ((w >> 8) & 0xF) * static_cast<int>(static_cast<std::int8_t>((a >> 8) & 0xFF));
    r += ((w >> 16) & 0xF) * static_cast<int>(static_cast<std::int8_t>((a >> 16) & 0xFF));
    r += ((w >> 24) & 0xF) * static_cast<int>(static_cast<std::int8_t>((a >> 24) & 0xFF));
    return r;
}

// 硬件路径：按 build 时探针选定的写法。NOVA_SYCL_DP4A==0/3 时等价于标量。
inline int d4_hw(std::int32_t w, std::int32_t a, int c) {
#if NOVA_SYCL_DP4A == 1
    return sycl::dp4a(as_i8x4(w), as_i8x4(a), c);
#elif NOVA_SYCL_DP4A == 2
    sycl::vec<std::int32_t, 1> r = sycl::dp4a(as_i8x4(w), as_i8x4(a), sycl::vec<std::int32_t, 1>(c));
    return r[0];
#else
    return d4_scalar(w, a, c);
#endif
}

// HW=true 走 d4_hw，false 走标量 —— 基准里"关掉 dp4a"就是实例化 <false>。
template <bool HW>
inline int d4(std::int32_t w, std::int32_t a, int c) {
    if constexpr (HW) {
        return d4_hw(w, a, c);
    } else {
        return d4_scalar(w, a, c);
    }
}

// ---- 16 B 装载 --------------------------------------------------------
// 用 memcpy 而不是 reinterpret_cast：clang 会把定长 memcpy 折成一条 128-bit 载入，
// 同时避免 strict-aliasing 的争论。
inline sycl::vec<std::int32_t, 4> load16(const void* p) {
    sycl::vec<std::int32_t, 4> v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
inline void store16(void* p, sycl::vec<std::int32_t, 4> v) { __builtin_memcpy(p, &v, sizeof(v)); }

// ---- subgroup 归约 ----------------------------------------------------
// 写成模板：体只在被实例化时检查，避免"没用到的 helper 把编译带崩"。
template <typename T>
inline T sg_sum(const sycl::sub_group& sg, T v) {
    return sycl::reduce_over_group(sg, v, sycl::plus<>());
}
template <typename T>
inline T sg_max(const sycl::sub_group& sg, T v) {
    return sycl::reduce_over_group(sg, v, sycl::maximum<>());
}
template <typename T>
inline T sg_bcast(const sycl::sub_group& sg, T v) {
    return sycl::group_broadcast(sg, v, std::size_t{0});
}

// ---- 一个 subgroup 量化一个 32 元素块 --------------------------------------
// Q4_K 的激活块是 32 个元素（每 16 个一个 int 和），而 subgroup 可以是 8/16/32，
// 所以每个 lane 要处理 PER = 32/SG 个元素。
// 2026-11-09 修：以前这里写死成「1 lane = 1 元素」，SG=16 时每个块只量化了前 16 个元素、
// s_hi 恒为 0（`sycl_bench` §8 的 quant 抽检抓到的：SG=16 时 int8 不一致 3327 个）。
template <int SG>
struct Q8Block {
    static constexpr int PER = 32 / SG;   // 每 lane 处理的元素数
    float scale;
    int   s_lo;                            // 元素 0..15 的和
    int   s_hi;                            // 元素 16..31 的和
};

// 输入本 lane 的 PER 个值（v），输出量化值（qi）+ 本块的 scale 与两个半块和。
// 所有 lane 都必须调用（内部走 subgroup 归约）。
template <int SG>
inline Q8Block<SG> q8_block(const sycl::sub_group& sg, int lane, const float* v, int* qi) {
    static_assert(32 % SG == 0, "subgroup 必须能整除 Q4_K 的 32 元素子块");
    float amax = 0.f;
    #pragma unroll
    for (int t = 0; t < Q8Block<SG>::PER; ++t) amax = sycl::fmax(amax, sycl::fabs(v[t]));
    amax = sg_max<float>(sg, amax);
    const float scale = amax * (1.f / 127.f);
    const float inv   = (amax > 0.f) ? (127.f / amax) : 0.f;
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int t = 0; t < Q8Block<SG>::PER; ++t) {
        qi[t] = static_cast<int>(sycl::rint(v[t] * inv));
        if (lane + t * SG < 16) s0 += qi[t];
        else                    s1 += qi[t];
    }
    Q8Block<SG> r;
    r.scale = scale;
    r.s_lo  = sg_sum<int>(sg, s0);
    r.s_hi  = sg_sum<int>(sg, s1);
    return r;
}


// ============================================================================
// 3. Q4_K 的 lane 映射
//    一个 subgroup(32) 吃一行；lane = 8 * blk_in_flight + c，c ∈ [0,8)。
//    c 决定：读 qs 的哪 16 B、对应哪 16 个 low-nibble 元素 / 16 个 high-nibble 元素。
//    与 src/kernel/q4k_gemv.cu 的 sb0/c/g/o/woff/alo/ahi/slo 完全同名同义。
// ============================================================================
struct Q4KLane {
    int block;     // 本 lane 负责的"第几个 super-block"（相对循环起点）
    int c;         // 0..7
    int g;         // 0..3：qs 的 32 B 组号
    int o;         // 0 / 16：组内字节偏移
    int woff;      // c * 16：相对 qs 的字节偏移
    int alo;       // 激活 low 侧偏移（64*g + o）
    int ahi;       // 激活 high 侧偏移（alo + 32）
    int slo;       // 子块号（2*g），high 侧用 slo+1
};

inline Q4KLane q4k_lane(int lane) {
    Q4KLane l;
    l.block = lane / 8;
    l.c     = lane % 8;
    l.g     = l.c >> 1;
    l.o     = (l.c & 1) << 4;
    l.woff  = l.c * 16;
    l.alo   = l.g * 64 + l.o;
    l.ahi   = l.alo + 32;
    l.slo   = 2 * l.g;
    return l;
}

// 一个 subgroup 一次并行覆盖几个 super-block
inline constexpr int kBlocksInFlight = syclb::kSyclSg / 8;

// ---- 方案 C：一行的 super-block 怎么分给 SPLITS 个 subgroup -----------------
// floor 切分：第 s 段 = [s*BPR/SPLITS, (s+1)*BPR/SPLITS)，段长最多差 1，段与段首尾相接，
// 4 个 subgroup 合起来正好覆盖整行，不重不漏。
//   1.5B（BPR=6, SPLITS=4）: 1/2/1/2        7B（BPR=14, SPLITS=4）: 3/4/3/4
// 与 q4k_gemv.cpp 里 kernel 用的是同一份算术（tests/sycl_gemv_host_check.cpp 直接跑这个 kernel）。
inline constexpr int q4k_seg_begin(int split, int splits, int bpr) { return split * bpr / splits; }
inline constexpr int q4k_seg_end(int split, int splits, int bpr) { return (split + 1) * bpr / splits; }

// ============================================================================
// 4. 常用 launch 形状
// ============================================================================
// 一维网格 + 固定 local：适合"每个 subgroup 一件活"的 kernel
inline sycl::nd_range<1> nd1(std::size_t groups, std::size_t local) {
    return sycl::nd_range<1>(sycl::range<1>(groups * local), sycl::range<1>(local));
}

} // namespace syclk
} // namespace nova
