// code/session.h — 会话：独立 KV 切片 + 独立工作区 + 独立 stream + 独立输出缓冲
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdio>
#include "map/qwen2.5/model_map.h"
#include "map/qwen2.5/tokenizer.h"
#include "kernel/1080ti/kernel_api.h"
#include "graph/kv_pool.h"

namespace nova {

inline std::mutex& console_mu() { static std::mutex m; return m; }

struct Msg { std::string role, content; };

struct Session {
    int      id       = -1;
    int64_t  n_past   = 0;
    bool     kv_valid = false;
    std::vector<Msg> history;
    KvSlice*         kslice    = nullptr;   // 块池里的独占切片
    int32_t*         table_dev = nullptr;   // 本会话的块表（设备，长度 nblocks）

    Workspace ws;              // 本会话独占的激活缓冲 + stream
    std::string obuf;          // 本会话输出缓冲（防串台）

    std::string      sys;        // 本会话的 system prompt
    std::string      label;      // 终端标签

    std::string      user_text;
    std::string      delta;
    int64_t          pos0 = 0;
    std::vector<int> prompt_ids;
    std::vector<int> out_ids;
    unsigned long long first_hash = 0; // 本对话首轮输入的哈希（块前缀）
    int                turn_no    = 0; // 轮次序号
    unsigned long long in_hash  = 0;   // 本轮的块前缀
    unsigned long long out_hash = 0;   // 本轮的块后缀

    std::string      pending;
    std::string      reply;
    double           prefill_ms = 0;
    double           decode_ms  = 0;

    const ModelCard* card   = nullptr;
    Tokenizer*       tk     = nullptr;
    Engine*          engine = nullptr;
    int              max_new = 96;
};

inline std::string render_full(const ModelCard& c, const std::vector<Msg>& h, const std::string& sys) {
    std::string p;
    if (c.asst) {                       // DeepSeek-R1 风格：<｜User｜>text<｜Assistant｜>
        for (const auto& m : h) {
            if (m.role == "user") { p += c.im_start; p += m.content; }
            else                  { p += c.asst;     p += m.content; p += c.im_end; }
        }
        p += c.asst;
        return p;
    }
    if (!sys.empty()) { p += c.im_start; p += "system\n"; p += sys; p += c.im_end; p += "\n"; }
    for (const auto& m : h) {
        p += c.im_start; p += m.role; p += "\n";
        p += m.content;
        p += c.im_end; p += "\n";
    }
    p += c.im_start; p += "assistant\n";
    return p;
}

class SessionManager {
public:
    void init(Engine* e) {
        engine_ = e;
        for (int i = 0; i < kMaxSessions; ++i) used_[i] = false;
        kv_pool().init(kMaxSessions, kMaxCtx);
        // 块表一次性放进 arena，按 slice 取偏移；每个会话一张 length=nblocks 的表
        kv_pool().set_dev(e->kv_table_slot(0));
    }

    Session* create(const ModelCard* card, Tokenizer* tk, int max_new) {
        std::lock_guard<std::mutex> lk(mu_);
        int slot = -1;
        for (int i = 0; i < kMaxSessions; ++i) if (!used_[i]) { slot = i; break; }
        if (slot < 0) { std::fprintf(stderr, "[ERROR] 会话已满 (%d)\n", kMaxSessions); std::exit(3); }
        used_[slot] = true;

        auto s = std::make_unique<Session>();
        s->id = slot; s->card = card; s->tk = tk; s->engine = engine_; s->max_new = max_new;

        NOVA_CU(cudaStreamCreate(&s->ws.s));
        engine_->make_workspace(&s->ws, kSessCtx, s->ws.s);

        // 独占切片 + 块表上设备
        s->kslice = kv_pool().acquire();
        if (!s->kslice) { std::fprintf(stderr, "[ERROR] KV 块池已空\n"); std::exit(3); }
        s->table_dev = s->kslice->table_dev;
        NOVA_CU(cudaMemcpyAsync(s->table_dev, s->kslice->table.data(),
                                (size_t)s->kslice->nblocks * sizeof(int32_t),
                                cudaMemcpyHostToDevice, s->ws.s));

        // 清零本会话自己的物理块：块池是共享的，绝不能整片 memset。
        // 第 2 步的块表是「切片内恒等」，所以这里按块逐个清（总量与改造前一致）。
        {
            const size_t blkf = (size_t)kKvBlock * (size_t)Engine::kv_width();
            for (int l = 0; l < Model::L; ++l) {
                const size_t loff = (size_t)l * (size_t)Engine::kv_nb();
                for (int b = 0; b < s->kslice->nblocks; ++b) {
                    const size_t off = (loff + (size_t)s->kslice->table[(size_t)b]) * blkf;
                    NOVA_CU(cudaMemsetAsync(engine_->kv_k_base() + off, 0, blkf * sizeof(float), s->ws.s));
                    NOVA_CU(cudaMemsetAsync(engine_->kv_v_base() + off, 0, blkf * sizeof(float), s->ws.s));
                }
            }
            NOVA_CU(cudaStreamSynchronize(s->ws.s));
        }

        Session* raw = s.get();
        live_.push_back(std::move(s));
        return raw;
    }

    Session* get(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : live_) if (s->id == id) return s.get();
        return nullptr;
    }

    void destroy(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = live_.begin(); it != live_.end(); ++it) {
            if ((*it)->id == id) {
                cudaStreamDestroy((*it)->ws.s);
                kv_pool().release((*it)->kslice);
                used_[id] = false;
                live_.erase(it);
                return;
            }
        }
    }

    int live_count() { std::lock_guard<std::mutex> lk(mu_); return (int)live_.size(); }

private:
    Engine* engine_ = nullptr;
    std::mutex mu_;
    bool    used_[kMaxSessions] = {false, false, false, false};
    std::vector<std::unique_ptr<Session>> live_;
};

inline SessionManager& sessions() { static SessionManager m; return m; }

} // namespace nova
