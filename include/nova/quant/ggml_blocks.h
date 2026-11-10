// nova_v2/include/nova/quant/ggml_blocks.h — ggml 的「小」量化块：Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q6_K
//
// 为什么需要：DeepSeek-V2-Lite 的 GGUF 是**混量化**（同一份文件里 Q4_K / Q5_0 / Q8_0 / Q6_K 都有，
// 见 docs/DEEPSEEK_V2_LITE.md §5.5），而本仓库原来只认 Q4_K（+ F32/F16/Q6_K/Q8_0 的尺寸）。
// 这些格式都是 **32 个元素一块**（Q6_K 是 256），块内只有一个（或两个）fp16 尺度，
// 所以反量化/点积都比 Q4_K 简单得多。
//
// 与 ggml 的逐字节对应（都按小端、fp16 在前）：
//   Q4_0 : { fp16 d;            uint8 qs[16] }   18 B   4-bit × 32，值 = d*(q-8)
//   Q4_1 : { fp16 d, m;         uint8 qs[16] }   20 B   4-bit × 32，值 = d*q+m
//   Q5_0 : { fp16 d;     uint8 qh[4]; uint8 qs[16] } 22 B  5-bit × 32，值 = d*(q-16)
//   Q5_1 : { fp16 d, m;  uint8 qh[4]; uint8 qs[16] } 24 B  5-bit × 32，值 = d*q+m
//   Q8_0 : { fp16 d;            int8  qs[32] }   34 B   值 = d*q
//   Q6_K : { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d } 210 B  6-bit × 256
//
// 约定：**这些块只在 host 侧用于反量化/对拍**；SYCL 侧的点积 kernel 另有实现
// （Q4_K 走 d4 那条路，其余要按同样思路补）。
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "nova/quant/q4_k.h"      // half_to_float / block_q4_K / kQK

namespace nova {

// ---------------------------------------------------------------------------
// 块结构（逐字节与 ggml 一致）
// ---------------------------------------------------------------------------
struct block_q4_0 { std::uint16_t d;            std::uint8_t qs[16]; };
struct block_q4_1 { std::uint16_t d, m;         std::uint8_t qs[16]; };
struct block_q5_0 { std::uint16_t d; std::uint8_t qh[4]; std::uint8_t qs[16]; };
struct block_q5_1 { std::uint16_t d, m;          std::uint8_t qh[4]; std::uint8_t qs[16]; };
struct block_q8_0 { std::uint16_t d;            std::int8_t  qs[32]; };
struct block_q6_K { std::uint8_t ql[128]; std::uint8_t qh[64]; std::int8_t scales[16]; std::uint16_t d; };

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 必须是 18 B");
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 必须是 20 B");
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 必须是 22 B");
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 必须是 24 B");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 必须是 34 B");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K 必须是 210 B");

// ---------------------------------------------------------------------------
// 反量化：单块 -> 32 / 256 个 float
// ---------------------------------------------------------------------------
inline void dequant_block_q4_0(const block_q4_0* b, float* y) {
    const float d = half_to_float(b->d);
    for (int j = 0; j < 16; ++j) {
        const int lo = (b->qs[j] & 0x0F) - 8;
        const int hi = (b->qs[j] >> 4)  - 8;
        y[j]      = d * static_cast<float>(lo);
        y[j + 16] = d * static_cast<float>(hi);
    }
}

inline void dequant_block_q4_1(const block_q4_1* b, float* y) {
    const float d = half_to_float(b->d);
    const float m = half_to_float(b->m);
    for (int j = 0; j < 16; ++j) {
        const int lo = b->qs[j] & 0x0F;
        const int hi = b->qs[j] >> 4;
        y[j]      = d * static_cast<float>(lo) + m;
        y[j + 16] = d * static_cast<float>(hi) + m;
    }
}

inline void dequant_block_q5_0(const block_q5_0* b, float* y) {
    const float d = half_to_float(b->d);
    // qh[4] = 32 个高位，按位序 j 取（bit j of little-endian qh）
    const std::uint32_t qh = static_cast<std::uint32_t>(b->qh[0]) |
                             (static_cast<std::uint32_t>(b->qh[1]) << 8) |
                             (static_cast<std::uint32_t>(b->qh[2]) << 16) |
                             (static_cast<std::uint32_t>(b->qh[3]) << 24);
    for (int j = 0; j < 16; ++j) {
        const int lo = ((b->qs[j] & 0x0F) | (((qh >> j)       & 1) << 4)) - 16;
        const int hi = ((b->qs[j] >> 4)   | (((qh >> (j + 16)) & 1) << 4)) - 16;
        y[j]      = d * static_cast<float>(lo);
        y[j + 16] = d * static_cast<float>(hi);
    }
}

inline void dequant_block_q5_1(const block_q5_1* b, float* y) {
    const float d = half_to_float(b->d);
    const float m = half_to_float(b->m);
    const std::uint32_t qh = static_cast<std::uint32_t>(b->qh[0]) |
                             (static_cast<std::uint32_t>(b->qh[1]) << 8) |
                             (static_cast<std::uint32_t>(b->qh[2]) << 16) |
                             (static_cast<std::uint32_t>(b->qh[3]) << 24);
    for (int j = 0; j < 16; ++j) {
        const int lo = (b->qs[j] & 0x0F) | (((qh >> j)        & 1) << 4);
        const int hi = (b->qs[j] >> 4)   | (((qh >> (j + 16)) & 1) << 4);
        y[j]      = d * static_cast<float>(lo) + m;
        y[j + 16] = d * static_cast<float>(hi) + m;
    }
}

inline void dequant_block_q8_0(const block_q8_0* b, float* y) {
    const float d = half_to_float(b->d);
    for (int j = 0; j < 32; ++j) y[j] = d * static_cast<float>(b->qs[j]);
}

// Q6_K：256 个元素 / 210 B。布局与 ggml dequantize_row_q6_K 逐行对应。
inline void dequant_block_q6_K(const block_q6_K* b, float* y) {
    const float d = half_to_float(b->d);
    const std::uint8_t* ql = b->ql;
    const std::uint8_t* qh = b->qh;
    const std::int8_t*  sc = b->scales;
    for (int n = 0; n < kQK; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = static_cast<int>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l + 0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0]  = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ---------------------------------------------------------------------------
// 一行按类型反量化 —— 由调用方（tools/quant_dump.cpp）按 GgmlType 分派，
// 本头文件不依赖 loader 的枚举，保持 quant 层独立。
// ---------------------------------------------------------------------------

} // namespace nova
