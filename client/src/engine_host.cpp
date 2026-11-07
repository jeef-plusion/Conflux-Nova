// engine_host.cpp — 真实接引擎：jthread 推理线程 + mutex 队列 + 零拷贝遥测
#include "engine_host.h"
#include "session.h"
#include "INPUT_CHAT.h"
#include "graph/kv_registry.h"

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace {
nova::Engine*    g_eng = nullptr;
nova::Tokenizer  g_tk;
nova::Session*   g_cur = nullptr;

std::mutex              g_mu;
std::queue<std::string> g_q;
std::atomic<bool>       g_busy{ false };
std::jthread            g_worker;

// 滑动平均吞吐
double g_tps = 0;
int    g_tok_cnt = 0;
std::chrono::steady_clock::time_point g_t0;
}

bool EngineHost::load(const std::string& gguf, std::string& err) {
    try {
        auto* e = new nova::Engine();
        e->load(gguf);
        nova::GgufFile meta(gguf);
        g_tk.load(meta);
        nova::sessions().init(e);
        g_eng = e;

        const nova::ModelCard* card = nova::kCards;
        g_cur = nova::sessions().create(card, &g_tk, 256);
        tree_.clear();
        tree_.push_back({ 0, "会话 1" });
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

void EngineHost::chat(const std::string& text) {
    if (!g_cur || g_busy.exchange(true)) return;

    g_worker = std::jthread([text] {
        nova::Session* s = g_cur;
        auto& e = *g_eng;

        // 1) 拼接 + 编码
        s->user_text = text;
        s->history.push_back({ "user", text });
        std::string delta = s->kv_valid
            ? std::string(s->card->im_end) + "\n" + s->card->im_start + "user\n" + text +
              s->card->im_end + "\n" + s->card->im_start + "assistant\n"
            : nova::render_full(*s->card, s->history, s->sys);
        const int64_t pos0 = s->kv_valid ? s->n_past : 0;
        auto ids = nova::encode_chat(g_tk, delta);

        // 2) 写入前 COW（Paged KV：共享块会在这一步分裂）
        if (pos0 > 0) nova::session_cow(s, pos0, (int)ids.size());

        // 3) prefill
        e.prefill(s->ws, ids.data(), (int)ids.size(), s->table_dev, pos0);
        s->n_past = pos0 + (int64_t)ids.size();
        s->kv_valid = true;

        // 4) 自回归 + 逐 token 入队
        g_t0 = std::chrono::steady_clock::now();
        g_tok_cnt = 0;
        int tok = e.sample_last(s->ws);
        const int im_end = g_tk.id(s->card->im_end, 151645);
        for (int i = 0; i < s->max_new && tok != im_end; ++i) {
            std::string piece = g_tk.decode(std::vector<int>{ tok });
            { std::lock_guard<std::mutex> lk(g_mu); g_q.push(piece); }
            ++g_tok_cnt;
            nova::session_cow(s, s->n_past, 1);
            tok = e.decode(s->ws, tok, s->table_dev, s->n_past);
            ++s->n_past;
        }

        // 5) 吞吐滑动平均
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
        if (dt > 0) g_tps = 0.7 * g_tps + 0.3 * (g_tok_cnt / dt);
        g_busy = false;
    });
}

void EngineHost::pump(const TokenFn& on_token) {
    for (;;) {
        std::string piece;
        { std::lock_guard<std::mutex> lk(g_mu);
          if (g_q.empty()) break;
          piece = std::move(g_q.front()); g_q.pop(); }
        on_token(piece);
    }
}

bool EngineHost::busy() const { return g_busy.load(); }

int64_t EngineHost::fork(int64_t block) {
    if (!g_cur) return -1;
    nova::Session* ns = nova::sessions().create(g_cur->card, &g_tk, g_cur->max_new);
    const int64_t tokens = (block + 1) * nova::kKvBlock;
    if (tokens > g_cur->n_past) return -1;
    nova::session_attach_prefix(ns, g_cur, tokens);   // ← refcnt++ 在这里发生
    tree_.push_back({ block, "Fork @" + std::to_string(block) });
    return block;
}

bool EngineHost::attach(int64_t block) {
    if (!g_cur || tree_.empty()) return false;
    const int64_t tokens = (block + 1) * nova::kKvBlock;
    if (tokens > g_cur->n_past) return false;
    nova::session_attach_prefix(g_cur, g_cur, tokens);  // 自并：仅演示语义
    return true;
}

Telemetry EngineHost::telemetry() const {
    Telemetry t;
    if (g_eng) {
        t.vram_used_gb = g_eng->arena_used_gb();
        t.vram_cap_gb  = g_eng->arena_cap_gb();
        t.blocks_total = nova::kv_pool().nb_total();
        t.blocks_free  = nova::kv_pool().free_blocks();
        int shared = 0;
        for (int b = 0; b < t.blocks_total; ++b)
            if (nova::kv_pool().refcnt(b) > 1) ++shared;
        t.blocks_shared = shared;
        t.cow_count = nova::kv_pool().cow_count();
    }
    if (g_cur) t.n_past = g_cur->n_past;
    t.tok_s = g_tps;
    return t;
}

const std::vector<ForkNode>& EngineHost::tree() const { return tree_; }