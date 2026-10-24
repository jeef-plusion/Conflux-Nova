// code/graph/qwen2.h — Qwen2 家族的层序列（dense：Qwen2.5-7B / 1.5B，及 R1-Distill-Qwen 7B / 1.5B 衍生变种）
//
// 家族差异只体现在「序列里有哪些算子」+「spec 里的 constexpr」，
// 执行器（dense.h）对所有 dense 家族是同一份。
#pragma once
#include "graph/dense.h"

namespace nova {
namespace graph {

template <class Spec>
struct Qwen2Layer {
    // Qwen2 系一律 Neox 布局（相邻半对）。bias 这一步留在序列里，
    // 是否真正生成代码由 Spec::has_qkv_bias 在编译期决定。
    static_assert(Spec::rope == RopeLayout::Neox, "qwen2 家族的 RoPE 布局必须是 Neox");

    using spec = Spec;
    using ops  = OpList<
        AttnNormOp,        // rms_norm(x, attn_norm) + quant
        QkvProjOp,         // q / k / v
        QkvBiasOp<Spec>,   // 三投影 bias（无 bias 的模具由 if constexpr 消掉）
        RopeKvOp,          // rope_batch + kv_store
        AttnOp,            // attn_decode / attn_prefill
        OProjOp,           // quant(attn_out) + o_proj
        ResidOp,           // x += attn_out
        FfnNormOp,         // rms_norm(x, ffn_norm) + quant
        GateUpOp,          // gate / up
        SwiGluOp,          // swiglu(gate, up) -> FFN 宽激活
        DownOp,            // down
        ResidOp>;          // x += ffn_out

    static constexpr const char* family = "qwen2-dense";
    static constexpr int         n_ops  = ops::size;

    // 执行 = 按 ops 的顺序展开成直线代码
    static void run(LayerCtx& c) { DenseLayer<Spec, ops>::run(c); }
};

// 家族路由：ModelSpec<Qwen2, ...> -> Qwen2Layer
template <class M>
struct GraphFor<M, Qwen2> : Qwen2Layer<GraphSpec<M>> {};

} // namespace graph
} // namespace nova
