// nova_v2/kernel.h — 模具：每个算子只对 ActiveModel 成立，形状全是编译期常量
// 没有任何函数指针、没有 plan 表、没有 ModelId 分支。换模型 = 重新编译。
#pragma once
#include <type_traits>
#include <cuda_runtime.h>
#include "nova/model/select.h"
#include "nova/kernel/types.h"

namespace nova {

using Model = ActiveModel;

// ---- 底层 launcher（定义在 src/kernel/*.cu，形状由模板参数钉死）----
void launch_rms_norm(float* y, const float* x, const float* w, int rows, cudaStream_t s);
void launch_add(float* acc, const float* x, int n, cudaStream_t s);
void launch_quant(ActQ8 dst, const float* x, int rows, cudaStream_t s);
void launch_quant_ffn(ActQ8 dst, const float* x, int rows, cudaStream_t s);
void launch_swiglu(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s);
void launch_swiglu_ffn(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s);

template <int K, int R>
void launch_gemv(const Q4KWeight& w, const ActQ8& x, float* out, cudaStream_t s);

template <int K, int R>
void mmq_launch(const Q4KWeight& w, const ActQ8& x, float* out, int n, cudaStream_t s);

// 编译期 K/R + 运行期列数：1 列走单列内核，其余按 32 列分块（权重只读一遍/块）
template <int K, int R>
void gemm_q4k(const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) {
    if (n == 1) { launch_gemv<K, R>(w, a, o, s); return; }
    mmq_launch<K, R>(w, a, o, n, s);
}

void launch_embed(float* x, const uint8_t* emb, const int* ids, int n, cudaStream_t s);
void launch_bias(float* y, const float* b, int n, int cols, cudaStream_t s);
void launch_kv_store(float* kc, float* vc, const float* k, const float* v, int n, int64_t pos0, cudaStream_t s);

void launch_rope_table(float* cs, float* sn, cudaStream_t s);
void launch_rope_batch(float* q, float* k, int n, int64_t pos0, const float* cs, const float* sn, cudaStream_t s);
void launch_rope(float* q, float* k, int64_t pos, const float* cs, const float* sn, cudaStream_t s);
void launch_attn_decode(const float* q, const float* kc, const float* vc, float* o,
                        int n_past, float* scratch, cudaStream_t s);
void launch_attn_prefill(const float* q, const float* kc, const float* vc, float* o,
                         int n_q, int n_past, cudaStream_t s);
void launch_sample(const float* logits, int* tok, cudaStream_t s);

// ---- 模型层：直接点名张量，没有"第几个投影"这种运行期信息 ----
struct Kernel {
    static constexpr int D   = Model::D;
    static constexpr int NH  = Model::NH;
    static constexpr int NKV = Model::NKV;
    static constexpr int HD  = Model::HD;
    static constexpr int FFN = Model::FFN;
    static constexpr int V   = Model::V;
    static constexpr int L   = Model::L;

    static void rms_norm(float* y, const float* x, const float* w, int rows, cudaStream_t s) {
        launch_rms_norm(y, x, w, rows, s);
    }
    static void residual(float* acc, const float* x, int n, cudaStream_t s) { launch_add(acc, x, n, s); }
    static void quant(ActQ8 dst, const float* x, int rows, cudaStream_t s) {
        launch_quant(dst, x, rows, s);
    }
    static void swiglu(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s) {
        launch_swiglu(dst, y, gate, up, rows, s);
    }
    static void swiglu_ffn(ActQ8 dst, float* y, const float* gate, const float* up, int rows, cudaStream_t s) {
        launch_swiglu_ffn(dst, y, gate, up, rows, s);
    }
    static void quant_ffn(ActQ8 dst, const float* x, int rows, cudaStream_t s) {
        launch_quant_ffn(dst, x, rows, s);
    }


    static void embed(float* x, const Q4KWeight& w, const int* ids, int n, cudaStream_t s) {
        launch_embed(x, w.blocks, ids, n, s);
    }
    static void bias(float* y, const float* b, int n, int cols, cudaStream_t s) { launch_bias(y, b, n, cols, s); }
    static void kv_store(float* kc, float* vc, const float* k, const float* v, int n, int64_t pos0, cudaStream_t s) {
        launch_kv_store(kc, vc, k, v, n, pos0, s);
    }
    // 任意列数（prefill 分块跑 32 列，decode 走单列）
    static void q_proj_n (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, NH * HD>(w, a, o, n, s); }
    static void k_proj_n (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, NKV * HD>(w, a, o, n, s); }
    static void v_proj_n (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, NKV * HD>(w, a, o, n, s); }
    static void o_proj_n (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<NH * HD, D>(w, a, o, n, s); }
    static void gate_n   (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, FFN>(w, a, o, n, s); }
    static void up_n     (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, FFN>(w, a, o, n, s); }
    static void down_n   (const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<FFN, D>(w, a, o, n, s); }
    static void lm_head_n(const Q4KWeight& w, const ActQ8& a, float* o, int n, cudaStream_t s) { gemm_q4k<D, V>(w, a, o, n, s); }

    static void rope_table(float* cs, float* sn, cudaStream_t s) { launch_rope_table(cs, sn, s); }
    static void rope_batch(float* q, float* k, int n, int64_t pos0, const float* cs, const float* sn, cudaStream_t s) {
        launch_rope_batch(q, k, n, pos0, cs, sn, s);
    }
    static void rope(float* q, float* k, int64_t pos, const float* cs, const float* sn, cudaStream_t s) {
        launch_rope(q, k, pos, cs, sn, s);
    }
    static void attn_decode(const float* q, const float* kc, const float* vc, float* o,
                            int n_past, float* scratch, cudaStream_t s) {
        launch_attn_decode(q, kc, vc, o, n_past, scratch, s);
    }
    static void attn_prefill(const float* q, const float* kc, const float* vc, float* o,
                             int n_q, int n_past, cudaStream_t s) {
        launch_attn_prefill(q, kc, vc, o, n_q, n_past, s);
    }
    static void sample(const float* logits, int* tok, cudaStream_t s) { launch_sample(logits, tok, s); }
};

} // namespace nova
