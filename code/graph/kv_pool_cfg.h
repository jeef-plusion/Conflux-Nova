// code/graph/kv_pool_cfg.h — KV 块大小：用户可指定，编译期定死
//
// 用法：
//   build_core.bat NOVA_MODEL_QWEN25_7B            -> 默认 16
//   nvcc ... -DNOVA_KV_BLOCK=32 ...                -> 32
//
// 为什么是编译期：块号与槽位靠移位和掩码算，移位量必须是常量，
// 否则每个 token 都要多一次分支或运行期除法——这正是我们要避免的东西。
#pragma once
#include <cstdint>

#ifndef NOVA_KV_BLOCK
#define NOVA_KV_BLOCK 16
#endif

namespace nova {

constexpr int kKvBlock = NOVA_KV_BLOCK;

static_assert(kKvBlock == 8 || kKvBlock == 16 || kKvBlock == 32 || kKvBlock == 64,
              "NOVA_KV_BLOCK 必须是 8 / 16 / 32 / 64 之一");

constexpr int kKvBlockShift = (kKvBlock == 8)  ? 3
                            : (kKvBlock == 16) ? 4
                            : (kKvBlock == 32) ? 5
                                               : 6;
constexpr int kKvBlockMask = kKvBlock - 1;

static_assert((1 << kKvBlockShift) == kKvBlock, "移位量与块大小不一致");

// 逻辑位置 -> (块表下标, 块内槽位)
inline int kv_block_index(int64_t pos) { return (int)(pos >> kKvBlockShift); }
inline int kv_slot_index (int64_t pos) { return (int)(pos &  kKvBlockMask); }

} // namespace nova
