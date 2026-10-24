// nova_v2/config.h — V2.0.0 铁律常量（唯一真源）
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace nova {

// 铁律 1：权重绝对上限 7 GiB。超了编译期炸，不在运行期降级。
inline constexpr std::size_t kWeightBudget = 7ull * 1024 * 1024 * 1024;

// 铁律 2：全部显存来自 init 时一次 cudaMalloc；算子内零分配。
inline constexpr std::size_t kArenaReserve = 768ull * 1024 * 1024;

// 铁律 3：只允许 Q4_K 一种块格式参与计算。
inline constexpr int kQK           = 256;
inline constexpr int kBlockBytes   = 144;
inline constexpr int kGroupsPerBlk = 8;    // 8 个 32-元素子块

// 编译期定死的运行参数（原 CN_* 环境变量）
inline constexpr int kCtx             = 4096;   // 上下文上限
inline constexpr int kSmemPerBlock    = 48 * 1024;
inline constexpr int kWarpsPerBlock   = 8;
inline constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
inline constexpr int kMaxBatch        = 4;

} // namespace nova

// 唯一允许的失败方式：断言/中止。没有 return false，没有 fallback。
#define NOVA_REQUIRE(cond) assert(cond)
#define NOVA_ABORT(msg)                                                       \
    do {                                                                      \
        std::fflush(stdout);                                                  \
        std::fprintf(stderr, "[nova] FATAL %s:%d  %s\n", __FILE__, __LINE__, msg); \
        std::fflush(stderr);                                                  \
        std::abort();                                                         \
    } while (0)
