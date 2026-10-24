// nova_v2/model.h — 权重常驻 + 逐层 forward（全显存内流转）
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "nova/kernel.h"
#include "nova/loader/gguf.h"
#include "nova/memory/arena.h"

namespace nova {

enum Phase { PH_EMBED = 0, PH_NORM, PH_MM, PH_ATTN, PH_ROPE, PH_ELEM, PH_HEAD, PH_COUNT };

struct Profiler {
    bool        on = false;
    cudaEvent_t e0{}, e1{};
    double      acc[PH_COUNT] = {0};
    int         cur = -1;

    void init() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    void begin(int p) {
        if (!on) return;
        cur = p;
        cudaEventRecord(e0);
    }
    void end() {
        if (!on || cur < 0) return;
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        acc[cur] += ms;
        cur = -1;
    }
};

struct LayerW {
    Q4KWeight q, k, v, o, gate, up, down;
    const float* qb = nullptr;
    const float* kb = nullptr;
    const float* vb = nullptr;
    const float* attn_norm = nullptr;
    const float* ffn_norm  = nullptr;
};

class Engine {
public:
    void load(const std::string& path, int max_tokens);
    void reset();
    void prefill(const int* ids, int n);
    int  decode(int id);

    float*  logits() const { return logits_; }
    int64_t n_past() const { return n_past_; }
    size_t  weight_bytes() const { return weight_bytes_; }

    Profiler prof;

private:
    void layer(int l, int n, int64_t pos0);
    void head(int n);

    DeviceArena    arena_;
    std::vector<LayerW> layers_;
    Q4KWeight      tok_embd_{};
    Q4KWeight      out_w_{};
    const float*   out_norm_ = nullptr;

    float *x_ = nullptr, *h_ = nullptr, *qb_ = nullptr, *kb_ = nullptr, *vb_ = nullptr;
    float *ab_ = nullptr, *gb_ = nullptr, *ub_ = nullptr, *db_ = nullptr;
    int8_t* act_ = nullptr;   float* actd_ = nullptr;   int* asum_ = nullptr;
    int8_t* act2_ = nullptr;  float* act2d_ = nullptr;  int* asum2_ = nullptr;
    float*  logits_ = nullptr;
    int*    ids_ = nullptr;
    const float *rcs_ = nullptr, *rsn_ = nullptr;
    float*  kv_k_ = nullptr;
    float*  kv_v_ = nullptr;
    float*  attn_scratch_ = nullptr;

    int64_t n_past_ = 0;
    size_t  weight_bytes_ = 0;
    int     n_max_ = 0;
    cudaStream_t s_ = nullptr;
};

} // namespace nova
