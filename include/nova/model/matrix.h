// nova_v2/model/matrix.h — 五个模型的静态形状 + Q4_K 权重预算（编译期算死）
#pragma once
#include <cstddef>
#include <type_traits>
#include "nova/config.h"
#include "nova/arch/traits.h"

namespace nova {

// 单个张量按 Q4_K 切块的字节数
constexpr std::size_t q4k_bytes(std::size_t n) { return ((n + kQK - 1) / kQK) * kBlockBytes; }

template <class Arch, int L_, int D_, int NH_, int NKV_, int HD_, int FFN_, int V_, bool TIE_, int THETA_>
struct ModelSpec {
    using arch = Arch;
    static constexpr int  L = L_, D = D_, NH = NH_, NKV = NKV_, HD = HD_, FFN = FFN_, V = V_;
    static constexpr bool tied  = TIE_;
    static constexpr int  theta = THETA_;
    static constexpr bool qkv_bias = ArchTraits<Arch>::kQKVBias;
    static constexpr float eps = std::is_same_v<Arch, Qwen2> ? 1e-6f : 1e-5f;
    static constexpr const char* arch_name = std::is_same_v<Arch, Qwen2> ? "qwen2" : "llama";

    // 每层：q, k, v, o, gate, up, down
    static constexpr std::size_t weight_bytes() {
        const std::size_t per =
              q4k_bytes((std::size_t)D * NH * HD)
            + q4k_bytes((std::size_t)D * NKV * HD) * 2
            + q4k_bytes((std::size_t)NH * HD * D)
            + q4k_bytes((std::size_t)D * FFN) * 3;
        return per * L + q4k_bytes((std::size_t)V * D) * (tied ? 1 : 2);
    }
    static constexpr std::size_t kv_bytes() {
        return 2ull * L * kCtx * NKV * HD * sizeof(float);
    }

    static_assert(D % 8 == 0 && V % 8 == 0 && FFN % 8 == 0, "dims must be 8-aligned (no tail guards)");
    static_assert(NH % NKV == 0, "GQA heads must divide");
    static_assert((NH * HD) % 8 == 0, "q/o rows must be 8-aligned");
    static_assert(HD == 128 || HD == 64, "only head_dim 64 or 128");
    static_assert(weight_bytes() <= kWeightBudget, "weights exceed the 7 GiB budget");
    static_assert(weight_bytes() + kv_bytes() + kArenaReserve <= 11ull * 1024 * 1024 * 1024,
                  "weights + KV + workspace exceed 11 GiB");
};

// 7B 系 GGUF 的词表按 152064 落盘（官方与 bartowski 都是），151936 是 tokenizer 原始条数
using Qwen25_7B   = ModelSpec<Qwen2, 28, 3584, 28, 4, 128, 18944, 152064, false, 1000000>;
using R1_Qwen7B   = ModelSpec<Qwen2, 28, 3584, 28, 4, 128, 18944, 152064, false, 1000000>;
// 1.5B：张量表里没有 output.weight（tie），但 GGUF 的 tie_word_embeddings 写成 0，以张量表为准
using Qwen25_1p5B = ModelSpec<Qwen2, 28, 1536, 12, 2, 128,  8960, 151936, true,  1000000>;
using R1_Qwen1p5B = ModelSpec<Qwen2, 28, 1536, 12, 2, 128,  8960, 151936, true,  1000000>;
// 变种：abliterated 微调把 theta 改回 1e4、output 解绑
using Abl1p5B     = ModelSpec<Qwen2, 28, 1536, 12, 2, 128,  8960, 151936, false, 10000>;
using R1_Llama8B  = ModelSpec<Llama, 32, 4096, 32, 8, 128, 14336, 128256, false,  500000>;

} // namespace nova
