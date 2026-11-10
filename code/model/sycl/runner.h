// nova_v2/code/model/sycl/runner.h — SYCL 端到端 decode（28 层串起来，逐 token）
//
// 对标 CUDA 版：code/graph/dense.h 的 12 个算子 + src/model.cpp 的 Engine::decode/head/sample_last。
// 差别只有三处（都是有意的）：
//   1. KV 用**连续布局** [L][ctx][NKV*HD]（不是 CUDA 的共享块池 + 块表）——
//      先把「能跑通」做出来，paged/COW 是下一步；
//   2. 单 token 逐次喂（decode 口径）。批量 prefill 走 attn_prefill 的分支还没接；
//   3. 采样在 host 上做（拉回 V 个 float，top-k/top-p，与 CUDA 版同一套参数），
//      因为 SYCL 侧还没有 sample kernel。这是第一条要优化的地方。
//
// 形状全部来自 nova/model/matrix.h 的 ActiveModel（-DNOVA_SYCL_MODEL_* 选），编译期焊死。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "device.h"
#include "kernels.h"
#include "nova/loader/gguf.h"

namespace nova {
namespace sycls {

// 算子层的一切（kD/kFFN/gemv/rmsnorm_quant/...）都在 nova::syclk 里；
// 这里统一引入，免得每处都写 syclk::（本命名空间只服务 runner，不对外）。
using namespace syclk;

// 一层的权重：7 个 Q4_K 矩阵 + 5 个 fp32 向量（两个 norm、q/k/v 的 bias）
struct LayerW {
    syclb::ChunkedWeight qkv, o, gate, up, down;   // q/k/v 在加载时拼成一块（QKV 融合）
    float* qb        = nullptr;
    float* kb        = nullptr;
    float* vb        = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm  = nullptr;
};

// 端到端 runner：只管「一次 decode 走完 28 层」，不碰 HTTP / 会话 / 分词之外的东西。
class E2ERunner {
public:
    // ctx_max：KV 的逻辑长度上限（<= nova::kCtx）
    explicit E2ERunner(const std::string& gguf_path, int ctx_max = 512);
    ~E2ERunner();

    E2ERunner(const E2ERunner&)            = delete;
    E2ERunner& operator=(const E2ERunner&) = delete;

    // 喂一个 token（位置 pos），返回「下一个 token」。
    // greedy=true 时走 argmax（可复现、便于对拍）；false 走 top-k/top-p/temp（同 CUDA 版）。
    int step(int token_id, int pos, bool greedy = true);

    // 预热 n 次（同一 token 反复跑，位置 0），用于提频 / 让 JIT 落地
    void warmup(int n);

    int    ctx_max()      const { return ctx_max_; }
    int    layers()       const { return kL; }
    double last_ms()      const { return last_ms_; }
    std::int64_t weight_bytes() const { return weight_bytes_; }
    int    last_token()   const { return last_tok_; }

    syclb::Device& device() { return dev_; }

private:
    void load_weights(const nova::GgufFile& g);
    void load_q4k(syclb::ChunkedWeight& w, const nova::GgufFile& g, const std::string& name, int rows, int cols);
    // QKV 融合：把 attn_q/k/v 三个 Q4_K 张量按行拼成一块 [kQKV + 2*kKV, kD]
    void load_qkv(syclb::ChunkedWeight& w, const nova::GgufFile& g, const std::string& prefix);
    void load_f32(float*& dst, const nova::GgufFile& g, const std::string& name, int n);

    void embed(int t);
    void layer(int l, int pos0);
    void head();
    int  sample(bool greedy);
    void check_ctx(int pos) const;

    syclb::Device dev_;
    sycl::queue&  q_;

    // 权重
    syclb::ChunkedWeight emb_;      // token_embd（同时当 lm_head：1.5B 是 tie）
    syclb::ChunkedWeight outw_;     // output.weight（非 tie 的模型才有）
    float* out_norm_ = nullptr;
    bool   tied_     = true;
    std::vector<LayerW> lw_;

    // 激活 / IO
    float* x_     = nullptr;
    float* h_     = nullptr;
    float* qb_    = nullptr;
    float* kb_    = nullptr;
    float* vb_    = nullptr;
    float* qkv_out_ = nullptr;   // 融合投影输出（q/k/v 三个指针都指进这块，零拷贝切分）
    float* ab_    = nullptr;
    float* gb_    = nullptr;
    float* ub_    = nullptr;
    float* db_    = nullptr;
    float* logits_ = nullptr;
    float* cs_    = nullptr;
    float* sn_    = nullptr;
    float* scratch_ = nullptr;
    float* kvk_   = nullptr;
    float* kvv_   = nullptr;
    int*   ids_   = nullptr;

    // ActQ8（D 宽 / FFN 宽）
    syclb::ActQ8View A_{};
    syclb::ActQ8View A2_{};

    std::int8_t*  aq_  = nullptr;
    float*        ad_  = nullptr;
    std::int32_t* as_  = nullptr;
    std::int8_t*  a2q_ = nullptr;
    float*        a2d_ = nullptr;
    std::int32_t* a2s_ = nullptr;

    std::vector<float> h_logits_;
    int      ctx_max_ = 512;
    double   last_ms_ = 0.0;
    int      last_tok_ = -1;
    std::int64_t weight_bytes_ = 0;
    std::uint64_t rng_ = 0x9E3779B97F4A7C15ull;
};

} // namespace sycls
} // namespace nova
