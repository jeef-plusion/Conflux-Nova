// nova_v2/model.h — 权重共享只读；工作区 + KV + Stream 按会话隔离
#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "nova/kernel.h"
#include "nova/loader/gguf.h"
#include "nova/memory/arena.h"
#include "graph/kv_pool.h"
#include "graph/kv_pool_cfg.h"   // kKvBlock：块池寻址的唯一真源
#include "model/sycl/scope_core.h"   // NovaScope 统计核心（纯 C++，CUDA/SYCL 共用）
#include "model/cuda/scope_cuda.h"   // NovaScope 的 CUDA 事件池与 pending 队列


namespace nova {

enum Phase { PH_EMBED = 0, PH_NORM, PH_MM, PH_ATTN, PH_ROPE, PH_ELEM, PH_HEAD, PH_COUNT };

struct Profiler {
    bool        on = false;
    cudaEvent_t e0{}, e1{};
    double      acc[PH_COUNT] = {0};
    int         cur = -1;

    // ---- NovaScope ----
    // scoped=true 时走非阻塞路径：begin 在流里记一个事件，end 再记一个，最后统一
    // resolve_all() 用 cudaEventElapsedTime 算时长。老路径（scoped=false）保持原样
    // （每段同步 + 累加 acc），供原有基准使用。
    bool        scoped = false;
    std::size_t sc_idx = 0;          // 当前这条在 Scope::recs() 里的下标
    cudaStream_t stream = nullptr;   // 本层用的流（dense.h 在 run() 开头设一次）
    cudaEvent_t  a_ = nullptr, b_ = nullptr;   // 当前 op 的事件对（来自 scope 的池）

    void init() { cudaEventCreate(&e0); cudaEventCreate(&e1); }

    void begin(int p) { begin(p, nullptr, 0); }

    // name/bytes 只有 scoped 时用得上；为空则回退成阶段名
    void begin(int p, const char* name, std::int64_t bytes) {
        if (scoped) {
            auto& s = nova::scope::Scope::get();
            sc_idx = s.recs().size();
            s.begin(name ? name : phase_tag(p), nova::scope::Phase::Compute, bytes);
            a_ = nova::scope::pool().take();
            b_ = nova::scope::pool().take();
            if (a_ && b_) cudaEventRecord(a_, stream);
            cur = p;
            return;
        }
        if (!on) return;
        cur = p;
        cudaEventRecord(e0);
    }

    void end() {
        if (scoped) {
            if (cur < 0) return;
            if (a_ && b_) {
                cudaEventRecord(b_, stream);
                nova::scope::pendings().push_back(nova::scope::PendingC{sc_idx, a_, b_});
            } else {
                auto& s = nova::scope::Scope::get();
                s.finish(s.recs()[sc_idx]);     // 没拿到事件：退回 host 时间
            }
            a_ = b_ = nullptr;
            cur = -1;
            return;
        }
        if (!on || cur < 0) return;
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        acc[cur] += ms; cur = -1;
    }

    static const char* phase_tag(int p) {
        switch (p) {
            case PH_EMBED: return "embed";
            case PH_NORM:  return "norm";
            case PH_MM:    return "mm";
            case PH_ATTN:  return "attn";
            case PH_ROPE:  return "rope";
            case PH_ELEM:  return "elem";
            case PH_HEAD:  return "head";
            default:       return "op";
        }
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

    // KV 是「共享块池 + 非连续寻址」：会话不再自带 KV 切片，只带一张块表。
    void prefill(Workspace& w, const int* ids, int n, const int32_t* table, int64_t pos0);
    int  decode(Workspace& w, int id, const int32_t* table, int64_t pos);
    int  sample_last(Workspace& w);

    // 写入前的 COW 守卫：pos 所在块若被共享，复制一份独占块并更新设备表
    void cow_guard(KvSlice& sl, int64_t pos, cudaStream_t s);

    struct BatchItem { int tok_in; const int32_t* table; int64_t pos; int tok_out; };
    void decode_batch(BatchItem* it, int n, cudaStream_t s);

    // ---- 块池视图（第 2 步：池由所有会话共享，寻址全靠 table）----
    float*   kv_k_base() const { return kv_k_pool_; }
    float*   kv_v_base() const { return kv_v_pool_; }
    int32_t* kv_table_slot(int s) const { return kv_table_ + (size_t)s * kv_nblocks(); }

    // 每 token 在每个池里的 float 数（= KVW，k/v 分池存放）
    static constexpr int kv_width() { return Model::NKV * Model::HD; }
    // 每会话的逻辑块数（>= kMaxCtx 的最小块数，kMaxCtx 是块大小整数倍）
    static constexpr int kv_nblocks() { return kMaxCtx / kKvBlock; }
    // 全局物理块数 NB
    static constexpr int kv_nb() { return kMaxSessions * kv_nblocks(); }
    static constexpr int64_t kv_span() { return (int64_t)Model::L * kMaxCtx * Model::NKV * Model::HD; }

    size_t weight_bytes() const { return weight_bytes_; }
    double arena_used_gb() const { return arena_.used() / 1073741824.0; }
    double arena_cap_gb()  const { return arena_.cap()  / 1073741824.0; }
    size_t workspace_bytes(int n_max) const { return ws_bytes(n_max); }

    Profiler prof;

private:
    void layer(Workspace& w, int l, int n, int64_t pos0, const int32_t* table);
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
    int *basum = nullptr, *basum2 = nullptr, *bids = nullptr;
    float* battn_scratch = nullptr;
    float*         kv_k_pool_ = nullptr;
    float*         kv_v_pool_ = nullptr;
    int32_t*       kv_table_  = nullptr;   // kMaxSessions * kv_nblocks() 个 int32，按会话切段
    std::vector<float> h_logits_;
    uint64_t       rng_ = 0x9E3779B97F4A7C15ull;
    size_t         weight_bytes_ = 0;
};

// 池的总量必须与「每会话切片 × 会话数」逐字节相同：改造只动寻址，不动一字节分配。
static_assert(kMaxCtx % kKvBlock == 0, "kMaxCtx 必须是块大小的整数倍，否则池会多出尾巴");
static_assert((int64_t)Model::L * Engine::kv_nb() * kKvBlock * Engine::kv_width()
                  == (int64_t)kMaxSessions * Engine::kv_span(),
              "块池总量与逐会话切片总量必须相等");

} // namespace nova
