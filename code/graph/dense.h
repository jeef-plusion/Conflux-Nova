// code/graph/dense.h — 通用 dense transformer 层执行器（qwen2.h / llama.h 共用）
//
// 一个算子 = 一个 struct，只做一件事；顺序由 OpList 决定，执行由逗号折叠展开成
// 直线代码（与重构前手写的 layer() 逐条对应，编译器看到的调用序列一模一样）。
#pragma once
#include <cstdint>
#include "graph/spec.h"
#include "nova/model.h"

namespace nova {
namespace graph {

// Q4_K 权重字节：每 256 个权重一个 144 B 的 super-block（NovaScope 用它算带宽）
inline constexpr std::int64_t KB4(std::int64_t rows, std::int64_t cols) {
    return rows * (cols / 256) * 144;
}

// 执行一层所需的全部上下文：本会话 workspace + 本层权重 + 共享 KV 块池 + 本会话块表
struct LayerCtx {
    Workspace&     w;
    const LayerW&  lw;
    Profiler&      prof;
    int            n;       // 列数：decode = 1，prefill = n
    int64_t        pos0;
    float*         pool_k;  // 共享块池基址：[layer][NB][KB][NKV][HD]
    float*         pool_v;
    const int32_t* table;   // 本会话块表（设备，长度 nblocks）
    int            layer;   // 当前层号
    int            nb;      // 全局物理块数 NB
    const float*   rcs;     // RoPE cos 表（引擎级只读，已按 theta 生成）
    const float*  rsn;     // RoPE sin 表
    ActQ8         A;       // D 宽激活量化暂存
    ActQ8         A2;      // FFN 宽激活量化暂存
};

// ---- 一层里的 12 个算子 ----

// attention 前的 rms_norm + 量化
struct AttnNormOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_NORM, "rmsnorm.attn", (std::int64_t)Kernel::D * 4);
        Kernel::rms_norm(c.w.h, c.w.x, c.lw.attn_norm, c.n, c.w.s);
        Kernel::quant(c.A, c.w.h, c.n, c.w.s);
        c.prof.end();
    }
};

// q / k / v 三投影（Q4_K：单列走 GEMV，多列走 MMQ）
struct QkvProjOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_MM, "gemv.q", KB4(c.lw.q.rows, c.lw.q.cols));
        Kernel::q_proj_n(c.lw.q, c.A, c.w.qb, c.n, c.w.s);
        Kernel::k_proj_n(c.lw.k, c.A, c.w.kb, c.n, c.w.s);
        Kernel::v_proj_n(c.lw.v, c.A, c.w.vb, c.n, c.w.s);
        c.prof.end();
    }
};

// q/k/v 三投影 bias：Qwen2 系有、LLaMA 系没有。
// 有无由 Spec::has_qkv_bias 在编译期决定，if constexpr 的假分支不生成任何代码。
template <class Spec>
struct QkvBiasOp {
    static void run(LayerCtx& c) {
        if constexpr (Spec::has_qkv_bias) {
            c.prof.begin(PH_ELEM, "bias.qkv", 0);
            Kernel::bias(c.w.qb, c.lw.qb, c.n * Spec::NH * Spec::HD, Spec::NH * Spec::HD, c.w.s);
            Kernel::bias(c.w.kb, c.lw.kb, c.n * Spec::NKV * Spec::HD, Spec::NKV * Spec::HD, c.w.s);
            Kernel::bias(c.w.vb, c.lw.vb, c.n * Spec::NKV * Spec::HD, Spec::NKV * Spec::HD, c.w.s);
            c.prof.end();
        }
    }
};

// RoPE（布局在 kernel 里由 ArchTraits::kRope 编译期选定）+ KV 落盘
struct RopeKvOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_ROPE, "rope+kv_store", (std::int64_t)c.n * Kernel::NKV * Kernel::HD * 2 * 4);
        Kernel::rope_batch(c.w.qb, c.w.kb, c.n, c.pos0, c.rcs, c.rsn, c.w.s);
        Kernel::kv_store_paged(c.pool_k, c.pool_v, c.table, c.layer, c.nb,
                               c.n, c.pos0, c.w.kb, c.w.vb, c.w.s);
        c.prof.end();
    }
};

// attention：单列走 decode 内核，多列走 prefill（n 是本次列数，不是架构差异）
struct AttnOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_ATTN, "attn_decode", (std::int64_t)(c.pos0 + 1) * Kernel::NKV * Kernel::HD * 2 * 4);
        if (c.n == 1) Kernel::attn_decode_paged(c.w.qb, c.pool_k, c.pool_v, c.table, c.layer, c.nb,
                                                (int)c.pos0 + 1, c.w.ab, c.w.attn_scratch, c.w.s);
        else          Kernel::attn_prefill_paged(c.w.qb, c.pool_k, c.pool_v, c.table, c.layer, c.nb,
                                                 c.n, (int)c.pos0, c.w.ab, c.w.s);
        c.prof.end();
    }
};

// o 投影前的量化 + o 投影
struct OProjOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_MM, "o_proj", KB4(c.lw.o.rows, c.lw.o.cols));
        Kernel::quant(c.A, c.w.ab, c.n, c.w.s);
        Kernel::o_proj_n(c.lw.o, c.A, c.w.db, c.n, c.w.s);
        c.prof.end();
    }
};

// 残差（attention 之后 / FFN 之后各一次，同一个算子）
struct ResidOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_ELEM, "residual", (std::int64_t)Kernel::D * 2 * 4);
        Kernel::residual(c.w.x, c.w.db, c.n * Kernel::D, c.w.s);
        c.prof.end();
    }
};

// FFN 前的 rms_norm + 量化
struct FfnNormOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_NORM, "rmsnorm.ffn", (std::int64_t)Kernel::D * 4);
        Kernel::rms_norm(c.w.h, c.w.x, c.lw.ffn_norm, c.n, c.w.s);
        Kernel::quant(c.A, c.w.h, c.n, c.w.s);
        c.prof.end();
    }
};

// gate / up
struct GateUpOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_MM, "gemv.gate", KB4(c.lw.gate.rows, c.lw.gate.cols));
        Kernel::gate_n(c.lw.gate, c.A, c.w.gb, c.n, c.w.s);
        Kernel::up_n(c.lw.up, c.A, c.w.ub, c.n, c.w.s);
        c.prof.end();
    }
};

// swiglu：gate 原地写回，同时量化成 FFN 宽激活
struct SwiGluOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_ELEM, "swiglu", (std::int64_t)Kernel::FFN * 4);
        Kernel::swiglu_ffn(c.A2, c.w.gb, c.w.gb, c.w.ub, c.n, c.w.s);
        c.prof.end();
    }
};

// down
struct DownOp {
    static void run(LayerCtx& c) {
        c.prof.begin(PH_MM, "gemv.down", KB4(c.lw.down.rows, c.lw.down.cols));
        Kernel::down_n(c.lw.down, c.A2, c.w.db, c.n, c.w.s);
        c.prof.end();
    }
};

// ---- 执行器：OpList -> 直线代码 ----
template <class Spec, class List>
struct DenseLayer;

template <class Spec, class... Ops>
struct DenseLayer<Spec, OpList<Ops...>> {
    static void run(LayerCtx& c) {
        static_assert(!Spec::moe, "dense.h 只执行 dense FFN 层，MoE 需要自己的执行器");
        (Ops::run(c), ...);   // 逗号折叠：严格从左到右，顺序 = OpList 的声明顺序
    }
};

} // namespace graph
} // namespace nova
