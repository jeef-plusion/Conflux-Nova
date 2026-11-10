// nova_v2/include/nova/model/deepseek_v2_lite.h — DeepSeek-V2-Lite 的形状（唯一真源）
//
// 与 model/matrix.h 的 dense 模型不同，V2-Lite 有两块现有执行器完全没有的东西：
//   1. **MLA**（Multi-head Latent Attention）：KV 不按 head 存，而是压缩成
//      kv_lora(512) + k_rope(64) = 576 个 float/token/layer；用的时候再经 kv_b 展开成
//      k_nope(每 head 128) 与 v(每 head 128)。
//   2. **MoE**：64 个 routed expert（top-6）+ 2 个 shared expert，expert FFN 宽 1408。
//
// 本头文件只放常量与 static_assert，不放代码：kernel 与测试都 include 它。
#pragma once

#include <cstddef>

#include "nova/config.h"
#include "nova/arch/traits.h"     // DeepSeekV2 标签的唯一定义点

namespace nova {

struct DeepSeekV2Lite {
    using arch = DeepSeekV2;

    // ---- 主干（与 HF config.json 一致）----
    static constexpr int   L     = 27;
    static constexpr int   D     = 2048;
    static constexpr int   NH    = 16;
    static constexpr int   NKV   = 16;      // MLA 里 K/V 每个 head 一套（展开后）
    static constexpr int   HD    = 128;     // = v_head_dim，也是 qk_nope 的宽
    static constexpr int   V     = 102400;
    static constexpr int   FFN   = 10944;   // 第 0 层（dense）的 FFN 宽
    static constexpr int   theta = 10000;
    static constexpr float eps   = 1e-6f;
    static constexpr bool  tied  = false;
    static constexpr bool  qkv_bias = false;
    static constexpr bool  moe   = true;

    // ---- MLA ----
    static constexpr int kKVLoraRank = 512;   // kv_a_proj 的压缩维
    static constexpr int kQKRopeDim  = 64;    // rope 维度（q/k 各带一份）
    static constexpr int kQKNopeDim  = 128;   // = HD，非 rope 部分
    static constexpr int kKVLatent   = kKVLoraRank + kQKRopeDim;   // 576：KV cache 每 token 每层
    static constexpr int kQPerHead   = kQKNopeDim + kQKRopeDim;    // 192：q / k 每个 head 的宽
    static constexpr int kVPerHead   = HD;                         // 128
    // kv_b 的输出行数：16 head × (128 nope + 128 v) = 4096
    static constexpr int kKVBOut     = NH * (kQKNopeDim + kVPerHead);

    // ---- MoE ----
    static constexpr int kNExpert    = 64;
    static constexpr int kTopK       = 6;
    static constexpr int kNShared    = 2;
    static constexpr int kExpertFFN  = 1408;
    // 第 0 层是 dense FFN，其余 26 层是 MoE
    static constexpr int kDenseFFNLayers = 1;

    // ---- 编译期约束（改常量时先把这里过一遍）----
    //
    // 注意：**V2-Lite 的两个 FFN 宽度都不是 256 的倍数**
    //   dense FFN  10944 = 256 × 42.75
    //   expert FFN  1408 = 256 × 5.5
    // 而 Q4_K 的一个 super-block 是 256 个元素，行宽必须是 256 的倍数 —— 也就是说
    // **现有「只支持 Q4_K」的引擎装不下这两个张量**。真要用 V2-Lite，必须二选一：
    //   (a) 给这两个张量加 Q4_0 / Q5_0 / F16 的支持（32 元素一块，1408 与 10944 都整除 32），
    //   (b) 把行宽 pad 到 256 的倍数（浪费 3%~12% 带宽，且 GGUF 里没有这种布局，得重排权重）。
    // 这条是 2026-11-10 写 MLA 时发现的，记在这里免得后面有人以为是笔误。
    static constexpr bool kFFN256Aligned  = (FFN % kQK == 0);         // false
    static constexpr bool kExp256Aligned  = (kExpertFFN % kQK == 0);  // false
    static constexpr bool kFFN32Aligned   = (FFN % 32 == 0);          // true
    static constexpr bool kExp32Aligned   = (kExpertFFN % 32 == 0);   // true

    static_assert(D % kQK == 0, "D 必须是 256 的倍数（Q4_K 无尾块）");
    static_assert(kFFN32Aligned && kExp32Aligned, "FFN 宽度至少要能被 32 整除（Q4_0/Q5_0 的块大小）");
    static_assert(NH == NKV, "MLA 展开后每 head 一套 K/V");
    static_assert(kQPerHead % 32 == 0, "q/k 每 head 的宽必须是 32 的倍数（subgroup 归约）");
    static_assert(kVPerHead % 32 == 0, "v 每 head 的宽必须是 32 的倍数");
    static_assert(kKVLoraRank % 256 == 0, "kv_lora 必须是 256 的倍数（Q4_K）");
    static_assert(kQKRopeDim == 64, "本实现按 64 维 rope 写死（表索引步长 2）");

    static constexpr int n_layers()  { return L; }
    static constexpr int n_moe()     { return L - kDenseFFNLayers; }

    // Q4_K 权重字节的量级检查：16B 参数 × 0.5625 B/参数 ≈ 9.0 GB。
    // 真实值以 GGUF 张量表为准（本仓库还没有这个文件）。
    static constexpr std::size_t weight_bytes_q4k() { return static_cast<std::size_t>(16.0e9 * 0.5625); }
};

} // namespace nova
