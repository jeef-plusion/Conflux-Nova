// nova_v2/model.h — 权重共享只读；工作区 + KV + Stream 按会话隔离
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
    void begin(int p) { if (!on) return; cur = p; cudaEventRecord(e0); }
    void end() {
        if (!on || cur < 0) return;
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        acc[cur] += ms; cur = -1;
    }
};

// 每会话一份：激活缓冲 + 流。引擎的权重不在这里。
struct Workspace {
    float *x = nullptr, *h = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr;
    float *ab = nullptr, *gb = nullptr, *ub = nullptr, *db = nullptr;
    int8_t *act = nullptr, *act2 = nullptr;
    float *actd = nullptr, *act2d = nullptr;
    int   *asum = nullptr, *asum2 = nullptr;
    float *logits = nullptr;
    float *attn_scratch = nullptr;
    int   *ids = nullptr;
    cudaStream_t s = nullptr;
    int n_max = 0;
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
    void load(const std::string& path);

    void make_workspace(Workspace* w, int n_max, cudaStream_t s);

    void prefill(Workspace& w, const int* ids, int n, float* kvk, float* kvv, int64_t pos0);
    int  decode(Workspace& w, int id, float* kvk, float* kvv, int64_t pos);
    int  sample_last(Workspace& w);

    struct BatchItem { int tok_in; float* kvk; float* kvv; int64_t pos; int tok_out; };
    void decode_batch(BatchItem* it, int n, cudaStream_t s);

    float* kv_k_slot(int s) const { return kv_k_pool_ + (size_t)s * kv_span(); }
    float* kv_v_slot(int s) const { return kv_v_pool_ + (size_t)s * kv_span(); }
    static constexpr int64_t kv_span() { return (int64_t)Model::L * kMaxCtx * Model::NKV * Model::HD; }

    size_t weight_bytes() const { return weight_bytes_; }
    size_t workspace_bytes(int n_max) const { return ws_bytes(n_max); }

    Profiler prof;

private:
    void layer(Workspace& w, int l, int n, int64_t pos0, float* kvk, float* kvv);
    void head(Workspace& w, int n);
    static size_t ws_bytes(int n);

    DeviceArena    arena_;
    std::vector<LayerW> layers_;
    Q4KWeight      tok_embd_{};
    Q4KWeight      out_w_{};
    const float*   out_norm_ = nullptr;
    float*         rcs_ = nullptr;
    float*         rsn_ = nullptr;
    // 批量 decode 的 4 行缓冲
    float *bx = nullptr, *bh = nullptr, *bq = nullptr, *bk = nullptr, *bv = nullptr, *bab = nullptr;
    float *bg = nullptr, *bu = nullptr, *bd = nullptr, *blogits = nullptr;
    int8_t *bact = nullptr, *bact2 = nullptr;
    float *bactd = nullptr, *bact2d = nullptr;
    int *basum = nullptr, *basum2 = nullptr, *bids = nullptr, *bpos = nullptr;
    float* battn_scratch = nullptr;
    float*         kv_k_pool_ = nullptr;
    float*         kv_v_pool_ = nullptr;
    std::vector<float> h_logits_;
    uint64_t       rng_ = 0x9E3779B97F4A7C15ull;
    size_t         weight_bytes_ = 0;
};

} // namespace nova
