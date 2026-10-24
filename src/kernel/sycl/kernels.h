// nova_v2/src/kernel/sycl/kernels.h — SYCL 算子层的唯一接口
//
// 与 include/nova/kernel.h 的 CUDA 版一一对应：同样的名字、同样的语义、同样的
// "形状编译期焊死"。差别只有三个：
//   1. stream -> sycl::queue；
//   2. 权重是 ChunkedWeight（每块 ≤1 GiB），launcher 内部按块循环，kernel 只见一块；
//   3. 多一个 dp4a 开关（模板参数 HW），供基准里"关掉 dp4a 再跑一次"。
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#include "sycl_common.h"
#include "device.h"

namespace nova {
namespace syclk {

using syclb::ActQ8View;
using syclb::ChunkedWeight;

// ---------------------------------------------------------------------------
// norm.cpp —— RMSNorm（可融合量化）
// ---------------------------------------------------------------------------
void rms_norm(float* y, const float* x, const float* w, int rows, sycl::queue& q);

// 融合版：y = rmsnorm(x)*w 的同时把 y 量化成 ActQ8。decode 每层省一次 dispatch。
void rmsnorm_quant(float* y, ActQ8View dst, const float* x, const float* w, int rows, sycl::queue& q);

// ---------------------------------------------------------------------------
// quant.cpp —— fp32 -> int8（每 32 一个 scale，每 16 一个和）
// ---------------------------------------------------------------------------
template <int K>
void quant_rows(ActQ8View dst, const float* x, int rows, sycl::queue& q);

// ---------------------------------------------------------------------------
// norm.cpp / elem.cpp —— 逐元素
// ---------------------------------------------------------------------------
void residual_add(float* acc, const float* x, int n, sycl::queue& q);
void bias_add(float* y, const float* b, int n, int cols, sycl::queue& q);

// silu(gate)*up -> y，同时量化。K = D（attention 后）或 FFN（FFN 中段）。
template <int K>
void swiglu_quant(ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q);

void swiglu      (ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q);  // K = D
void swiglu_ffn  (ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q);  // K = FFN

// ---------------------------------------------------------------------------
// elem.cpp —— embed / KV 写入
// ---------------------------------------------------------------------------
void embed_rows(float* x, const std::uint8_t* emb_q4k, const int* ids, int n, sycl::queue& q);
void kv_store(float* kc, float* vc, const float* k, const float* v, int n, std::int64_t pos0, sycl::queue& q);

// ---------------------------------------------------------------------------
// rope.cpp
// ---------------------------------------------------------------------------
void rope_table(float* cs, float* sn, sycl::queue& q);
void rope_apply(float* qv, float* kv, std::int64_t pos, const float* cs, const float* sn, sycl::queue& q);
void rope_apply_batch(float* qv, float* kv, int n, std::int64_t pos0,
                      const float* cs, const float* sn, sycl::queue& q);

// ---------------------------------------------------------------------------
// q4k_gemv.cpp —— decode 主路径：Q4_K 权重 x int8 激活
//   K = 每行的列数（256 的倍数），行数由 ChunkedWeight::rows() 给出。
//   HW = true 走硬件 dp4a（若编译期可用），false 强制标量（基准对拍用）。
// ---------------------------------------------------------------------------
template <int K, bool HW>
void gemv(const ChunkedWeight& w, const ActQ8View& a, float* out, sycl::queue& q);

// ---------------------------------------------------------------------------
// q4k_gemv4.cpp —— 四列批量 GEMV：权重 16 B 只读一次，喂 4 组激活
//   激活按列主序拼在一起：xq4 + c*K / xd4 + c*(K/32) / xs4 + c*(K/16)
//   输出 out4 + c*rows + row
// ---------------------------------------------------------------------------
template <int K, bool HW>
void gemv4(const ChunkedWeight& w, const std::int8_t* xq4, const float* xd4, const std::int32_t* xs4,
           float* out4, sycl::queue& q);

// ---------------------------------------------------------------------------
// attn.cpp
// ---------------------------------------------------------------------------
// n_past 的语义与 CUDA 版一致：KV 里已有 n_past+1 个位置（含当前 token）。
void attn_decode(const float* q, const float* kc, const float* vc, float* o,
                 int n_past, float* scratch, sycl::queue& q);

// dp4a 指令级微基准：纯寄存器循环，8 条 dp4a/迭代（标量版同样的 32 个乘加）。
// hw=false 强制标量。用于回答"dp4a 是否真的落在一条指令上"（E1）。
void dp4a_micro(bool hw, int iters, int work_items, std::int32_t* out, sycl::queue& q);

// 需要的外部 scratch（float 个数）：调用方按 max_ctx 一次分配
constexpr int attn_decode_scratch_floats() {
    return kNH * kAttnSplit * (kHD + 2);
}

void attn_prefill(const float* q, const float* kc, const float* vc, float* o,
                  int n_q, int n_past, sycl::queue& q);

// ---------------------------------------------------------------------------
// 显式实例化清单（定义在各自 .cpp 里）；漏一个就是链接错误，不会静默走慢路径。
// ---------------------------------------------------------------------------
extern template void quant_rows<kD   >(ActQ8View, const float*, int, sycl::queue&);
extern template void quant_rows<kFFN >(ActQ8View, const float*, int, sycl::queue&);
extern template void swiglu_quant<kD >(ActQ8View, float*, const float*, const float*, int, sycl::queue&);
extern template void swiglu_quant<kFFN>(ActQ8View, float*, const float*, const float*, int, sycl::queue&);
extern template void gemv <kD,   true >(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);
extern template void gemv <kD,   false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);
extern template void gemv <kFFN, true >(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);
extern template void gemv <kFFN, false>(const ChunkedWeight&, const ActQ8View&, float*, sycl::queue&);
extern template void gemv4<kD,   true >(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&);
extern template void gemv4<kD,   false>(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&);
extern template void gemv4<kFFN, true >(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&);
extern template void gemv4<kFFN, false>(const ChunkedWeight&, const std::int8_t*, const float*, const std::int32_t*, float*, sycl::queue&);

} // namespace syclk
} // namespace nova
