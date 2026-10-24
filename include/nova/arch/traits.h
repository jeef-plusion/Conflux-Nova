// nova_v2/arch/traits.h — Qwen2 / Llama 的编译期差异（没有任何运行期 if）
#pragma once

namespace nova {

struct Qwen2 {};   // Qwen2.5 全系 + DeepSeek-R1-Distill-Qwen
struct Llama {};   // DeepSeek-R1-Distill-Llama-8B (Llama-3.1 结构)

enum class RopeLayout { Norm, Neox };       // Norm: 相邻成对；Neox: 前后半对
enum class MaskKind   { Causal, CausalSWA }; // SWA 只有 Qwen2 变体用，模板参数化

template <class A> struct ArchTraits;

template <> struct ArchTraits<Qwen2> {
    using arch = Qwen2;
    static constexpr RopeLayout kRope    = RopeLayout::Neox;
    static constexpr MaskKind   kMask    = MaskKind::Causal;
    static constexpr bool       kQKVBias = true;    // q/k/v 三投影带 bias
    static constexpr bool       kTiedEmb = true;    // 1.5B 系 tie，7B 系由 ModelSpec 覆盖
};

template <> struct ArchTraits<Llama> {
    using arch = Llama;
    static constexpr RopeLayout kRope    = RopeLayout::Norm;
    static constexpr MaskKind   kMask    = MaskKind::Causal;
    static constexpr bool       kQKVBias = false;
    static constexpr bool       kTiedEmb = false;
};

} // namespace nova
