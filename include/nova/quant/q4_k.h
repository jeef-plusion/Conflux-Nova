// nova_v2/quant/q4_k.h — Q4_K 块布局 + 反量化（与 ggml block_q4_K 字节级一致）
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include "nova/config.h"
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

// 主机端 selftest 也要能编，nvcc 之外不展开 __host__/__device__
#if defined(__CUDACC__)
#define NOVA_HD __host__ __device__
#else
#define NOVA_HD
#endif

namespace nova {

// 144 B / 256 权重
struct block_q4_K {
    uint16_t d;          // fp16,  super-block 尺度
    uint16_t dmin;       // fp16,  super-block 最小值的尺度
    uint8_t  scales[12]; // 8 个 32-子块的 6-bit (scale, min) 共 96 bit，压成 12 B
    uint8_t  qs[128];    // 4 x 32 B，低 nibble = 前 32 元素，高 nibble = 后 32 元素
};
static_assert(sizeof(block_q4_K) == kBlockBytes, "block_q4_K must be 144 bytes");

// fp16 -> fp32（设备端直接 __half2float；主机端软件展开，供 selftest 用）
inline NOVA_HD float half_to_float(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(*reinterpret_cast<const __half*>(&h));
#else
    const int      s = (h >> 15) & 1;
    const int      e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FFu;
    double v;
    if (e == 0)       v = std::ldexp((double)m, -24);          // 0 与次正规
    else if (e == 31) v = (m != 0) ? std::numeric_limits<double>::quiet_NaN()
                                    : std::numeric_limits<double>::infinity();
    else              v = std::ldexp((double)(m | 0x400u), e - 25);
    return (float)(s ? -v : v);
#endif
}

// 第 j 个子块（0..7）的 6-bit (scale, min)：前 4 组直接取低 6 位，后 4 组由高 2 位拼出
inline NOVA_HD void get_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m  = (uint8_t)((q[j + 4] >> 4)  | ((q[j]     >> 6) << 4));
    }
}

// 反量化一整块到 y[0..255]
inline NOVA_HD void dequant_block(const block_q4_K* b, float* y) {
    const float d    = half_to_float(b->d);
    const float dmin = half_to_float(b->dmin);
    const uint8_t* q = b->qs;
    // 8 个 32-元素子块各有独立的 6-bit (scale,min)：低 nibble 用 2g，高 nibble 用 2g+1
    for (int g = 0; g < 4; ++g) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(2 * g,     b->scales, sc0, m0);
        get_scale_min_k4(2 * g + 1, b->scales, sc1, m1);
        const float d0 = d * (float)sc0, e0 = dmin * (float)m0;
        const float d1 = d * (float)sc1, e1 = dmin * (float)m1;
        for (int l = 0; l < 32; ++l) y[g * 64 + l]      = d0 * (float)(q[l] & 0xF) - e0;
        for (int l = 0; l < 32; ++l) y[g * 64 + 32 + l] = d1 * (float)(q[l] >> 4)  - e1;
        q += 32;
    }
}

// 反量化一行（rows 行的第 r 行）到 y[cols]
inline void dequant_row(const uint8_t* blocks, int64_t r, int64_t cols, float* y) {
    const block_q4_K* row = reinterpret_cast<const block_q4_K*>(blocks) + r * (cols / kQK);
    for (int64_t b = 0; b < cols / kQK; ++b) dequant_block(row + b, y + b * kQK);
}

} // namespace nova
