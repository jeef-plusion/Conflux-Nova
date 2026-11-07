// engine_host.cpp — 真实接引擎：jthread 推理线程 + mutex 队列 + 零拷贝遥测
#include "engine_host.h"
#include "session.h"
#include "INPUT_CHAT.h"
#include "graph/kv_registry.h"
#include "nova/model/select.h"

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cmath>

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
        FILE* fp = std::fopen(gguf.c_str(), "rb");
        if (!fp) { err = "文件不存在: " + gguf; return false; }
        std::fclose(fp);

        // 先做「软校验」：verify_model 不匹配会直接 abort，UI 会连进程一起死。
        // 这里在调用引擎之前自己比对关键字段，给出可读的错误。
        {
            std::fprintf(stderr, "[client] precheck open\n"); std::fflush(stderr);
            nova::GgufFile g0(gguf);
            std::fprintf(stderr, "[client] precheck parsed\n"); std::fflush(stderr);
            const std::string arch = g0.s("general.architecture");
            auto hasu = [&](const char* k) { return g0.has(arch + k); };
            const uint64_t bc = hasu(".block_count")        ? g0.u(arch + ".block_count") : 0;
            const uint64_t el = hasu(".embedding_length")   ? g0.u(arch + ".embedding_length") : 0;
            const uint64_t nh = hasu(".attention.head_count") ? g0.u(arch + ".attention.head_count") : 0;
            // rope_theta 在不同 GGUF 里可能是 .rope_theta 或 .rope.freq_base
            double th = 0;
            if (g0.has(arch + ".rope_theta"))          th = g0.f(arch + ".rope_theta");
            else if (g0.has(arch + ".rope.freq_base")) th = g0.f(arch + ".rope.freq_base");
            char b[256];
            if (bc && bc != (uint64_t)nova::Model::L) {
                std::snprintf(b, sizeof(b), "层数不匹配：文件 %llu，本引擎 %d（需重新编译）",
                              (unsigned long long)bc, (int)nova::Model::L);
                err = b; return false;
            }
            if (el && el != (uint64_t)nova::Model::D) {
                std::snprintf(b, sizeof(b), "隐藏维不匹配：文件 %llu，本引擎 %d",
                              (unsigned long long)el, (int)nova::Model::D);
                err = b; return false;
            }
            if (nh && nh != (uint64_t)nova::Model::NH) {
                std::snprintf(b, sizeof(b), "注意力头数不匹配：文件 %llu，本引擎 %d",
                              (unsigned long long)nh, (int)nova::Model::NH);
                err = b; return false;
            }
            std::fprintf(stderr, "[client] precheck bc=%llu el=%llu nh=%llu th=%.1f mold_theta=%d\n",
                         (unsigned long long)bc, (unsigned long long)el, (unsigned long long)nh, th, (int)nova::Model::theta);
            std::fflush(stderr);
            if (th > 0 && std::fabs(th - (double)nova::Model::theta) > 1.0) {
                std::snprintf(b, sizeof(b), "rope_theta 不匹配：文件 %.0f，本引擎 %d —— 这是另一个模型变体",
                              th, (int)nova::Model::theta);
                err = b; return false;
            }
        }
        auto* e = new nova::Engine();
        e->load(gguf);
        nova::GgufFile meta(gguf);
        g_tk.load(meta);
        nova::sessions().init(e);
        g_eng = e;

        // 按编译期模具名选卡（一个二进制 = 一个模型），再用别名兜底
        const nova::ModelCard* card = nullptr;
        for (int i = 0; i < nova::kCardCount && !card; ++i)
            if (std::strcmp(nova::kCards[i].name, kModelName) == 0) card = &nova::kCards[i];
        if (!card && kModelAliases[0])
            for (int i = 0; i < nova::kCardCount && !card; ++i)
                if (std::strcmp(nova::kCards[i].name, kModelAliases[0]) == 0) card = &nova::kCards[i];
        if (!card) card = nova::kCards;

        std::printf("[client] card = %s\n", card->name);
        g_cur = nova::sessions().create(card, &g_tk, 256);
        tree_.clear();
        tree_.push_back({ -1, std::string("当前会话 · ") + card->name });
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

void EngineHost::chat(const std::string& text) {
    if (!g_cur || g_busy.exchange(true)) return;

    g_worker = std::jthread([this, text] {
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
        const int64_t tail_block = (s->n_past - 1) / nova::kKvBlock;
        tree_.push_back({ tail_block, "轮次 @" + std::to_string(tail_block) });
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
    // Fork 到「该块末尾」与「当前已生成」的较小者：块可能只有半满
    int64_t tokens = (block + 1) * (int64_t)nova::kKvBlock;
    if (tokens > g_cur->n_past) tokens = g_cur->n_past;
    if (tokens <= 0) return -1;
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