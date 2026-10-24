// nova_v2/kernel/types.h — 统一黑盒调用规范：只有指针 + 形状 + stream
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "nova/config.h"

namespace nova {

// 稠密 fp32 行主序
struct TensorView {
    float*  p;
    int64_t rows;
    int64_t cols;
};

// Q4_K 权重：rows 行、cols 列，cols 必须是 256 的倍数
struct Q4KWeight {
    const uint8_t* blocks;   // rows * (cols/256) 个 block_q4_K，行主序
    int64_t        rows;
    int64_t        cols;
};

// Q8 激活：每 32 元素一个 fp32 scale（qs 行内连续，16B 对齐）
struct ActQ8 {
    const int8_t* qs;   // rows * cols
    const float*  d;    // rows * (cols/32)
    const int*    s;    // rows * (cols/32)：每个 32-块的 int8 和（省掉 GEMV 里的 2 条 dp4a）
};

// 每层形状：编译期由 model/matrix.h 填，运行期只读
struct LayerShape {
    int d;        // hidden
    int nh;       // q heads
    int nkv;      // kv heads
    int hd;       // head_dim
    int ffn;
    int n_ctx;
};

} // namespace nova
