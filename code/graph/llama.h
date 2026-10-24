// code/graph/llama.h — Llama 家族（R1-Distill-Llama-8B：32 层 / D4096 / 32:8 / ffn14336 / V128256 / theta 5e5）
//
// 与 Qwen2 的唯一层结构差异：q/k/v 三投影没有 bias，所以序列里就没有 bias 这一步；
// 另外 RoPE 是 Norm 布局（相邻成对）。两者都在编译期定死。
#pragma once
#include "graph/dense.h"

namespace nova {
namespace graph {

template <class Spec>
struct LlamaLayer {
    static_assert(Spec::rope == RopeLayout::Norm, "llama 家族的 RoPE 布局必须是 Norm");
    static_assert(!Spec::has_qkv_bias, "llama 家族的层序列里没有 QKV bias 这一步");

    using spec = Spec;
    using ops  = OpList<
        AttnNormOp,        // rms_norm(x, attn_norm) + quant
        QkvProjOp,         // q / k / v（无 bias）
        RopeKvOp,          // rope_batch + kv_store
        AttnOp,            // attn_decode / attn_prefill
        OProjOp,           // quant(attn_out) + o_proj
        ResidOp,           // x += attn_out
        FfnNormOp,         // rms_norm(x, ffn_norm) + quant
        GateUpOp,          // gate / up
        SwiGluOp,          // swiglu(gate, up) -> FFN 宽激活
        DownOp,            // down
        ResidOp>;          // x += ffn_out

    static constexpr const char* family = "llama-dense";
    static constexpr int         n_ops  = ops::size;

    // 执行 = 按 ops 的顺序展开成直线代码
    static void run(LayerCtx& c) { DenseLayer<Spec, ops>::run(c); }
};

// 家族路由：ModelSpec<Llama, ...> -> LlamaLayer
template <class M>
struct GraphFor<M, Llama> : LlamaLayer<GraphSpec<M>> {};

} // namespace graph
} // namespace nova
