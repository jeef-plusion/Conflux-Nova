// code/graph/spec.h — 计算图层：由 ModelSpec 派生的静态层描述
//
// 这一层只做一件事：把「这个模型的一层由哪些算子、按什么顺序组成」变成编译期常量。
// 架构差异（QKV bias / RoPE 布局 / tie / MoE）全部是 constexpr，下游用 if constexpr
// 或模板特化消化 —— 推理热路径里没有 if (has_bias) 这种运行期分支。
#pragma once
#include <type_traits>
#include "nova/model/matrix.h"
#include "nova/arch/traits.h"

namespace nova {
namespace graph {

// 编译期算子序列：严格按声明顺序执行，零运行期开销（没有 switch、没有 plan 表）
template <class... Ops>
struct OpList {
    static constexpr int size = (int)sizeof...(Ops);
};

// ModelSpec -> 层描述。全部字段都是 constexpr，运行期不占一个字节。
template <class M>
struct GraphSpec {
    using model  = M;
    using arch   = typename M::arch;
    using traits = ArchTraits<arch>;

    static constexpr int   L = M::L, D = M::D, NH = M::NH, NKV = M::NKV, HD = M::HD,
                           FFN = M::FFN, V = M::V;
    static constexpr int   theta = M::theta;
    static constexpr float eps   = M::eps;

    // ---- 架构差异：只有这些 constexpr 能区分模型 ----
    static constexpr bool       has_qkv_bias = M::qkv_bias;   // = ArchTraits<arch>::kQKVBias
    static constexpr RopeLayout rope         = traits::kRope; // Neox / Norm
    static constexpr MaskKind   mask         = traits::kMask; // Causal / CausalSWA
    static constexpr bool       tied         = M::tied;
    static constexpr bool       moe          = false;         // 本执行器只支持 dense FFN
    static constexpr int        n_expert     = 1;

    static constexpr const char* arch_name = M::arch_name;
};

// 家族 × 模型 -> 层执行器；由家族头文件（qwen2.h / llama.h）特化。
template <class M, class A = typename M::arch>
struct GraphFor;

} // namespace graph
} // namespace nova
