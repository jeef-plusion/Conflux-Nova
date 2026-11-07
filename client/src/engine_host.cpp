// engine_host.cpp — 多轮上下文：增量 ChatML + KV 续接，不重算历史
#include "engine_host.h"
#include "session.h"
#include "graph/kv_registry.h"
#include "nova/model/select.h"
#include "INPUT_CHAT.h"

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <map>

namespace {
nova::Engine*   g_eng = nullptr;
nova::Tokenizer g_tk;

struct Sess {
    nova::Session* hw = nullptr;
    std::vector<Msg> hist;
    std::string title = "新会话";
    int turns = 0;
};
std::map<std::string, Sess> g_sess;
std::string g_last;
int g_seq = 0;

std::mutex              g_mu;
std::queue<std::string> g_q;
std::atomic<bool>       g_busy{ false };
std::jthread            g_worker;
std::string             g_cur_sid;
double g_tps = 0;
}

bool EngineHost::load(const std::string& gguf, std::string& err) {
    try {
        FILE* fp = std::fopen(gguf.c_str(), "rb");
        if (!fp) { err = "文件不存在: " + gguf; return false; }
        std::fclose(fp);

        {
            nova::GgufFile g0(gguf);
            const std::string arch = g0.s("general.architecture");
            auto hasu = [&](const char* k) { return g0.has(arch + k); };
            const uint64_t bc = hasu(".block_count") ? g0.u(arch + ".block_count") : 0;
            const uint64_t el = hasu(".embedding_length") ? g0.u(arch + ".embedding_length") : 0;
            const uint64_t nh = hasu(".attention.head_count") ? g0.u(arch + ".attention.head_count") : 0;
            double th = 0;
            if (g0.has(arch + ".rope_theta")) th = g0.f(arch + ".rope_theta");
            else if (g0.has(arch + ".rope.freq_base")) th = g0.f(arch + ".rope.freq_base");
            char b[256];
            if (bc && bc != (uint64_t)nova::Model::L) { std::snprintf(b, sizeof(b), "层数不匹配：文件 %llu，引擎 %d", (unsigned long long)bc, (int)nova::Model::L); err = b; return false; }
            if (el && el != (uint64_t)nova::Model::D) { std::snprintf(b, sizeof(b), "隐藏维不匹配：文件 %llu，引擎 %d", (unsigned long long)el, (int)nova::Model::D); err = b; return false; }
            if (nh && nh != (uint64_t)nova::Model::NH) { std::snprintf(b, sizeof(b), "注意力头数不匹配：文件 %llu，引擎 %d", (unsigned long long)nh, (int)nova::Model::NH); err = b; return false; }
            if (th > 0 && std::fabs(th - (double)nova::Model::theta) > 1.0) {
                std::snprintf(b, sizeof(b), "rope_theta 不匹配：文件 %.0f，引擎 %d", th, (int)nova::Model::theta);
                err = b; return false;
            }
        }

        auto* e = new nova::Engine();
        e->load(gguf);
        nova::GgufFile meta(gguf);
        g_tk.load(meta);
        nova::sessions().init(e);
        g_eng = e;
        g_sess.clear();
        g_last.clear();
        return true;
    } catch (const std::exception& ex) { err = ex.what(); return false; }
}

std::string EngineHost::new_session() {
    if (!g_eng) return {};
    const nova::ModelCard* card = nullptr;
    for (int i = 0; i < nova::kCardCount && !card; ++i)
        if (std::strcmp(nova::kCards[i].name, kModelName) == 0) card = &nova::kCards[i];
    if (!card) card = nova::kCards;

    nova::Session* s = nova::sessions().create(card, &g_tk, 512);
    if (!s) return {};
    char id[32]; std::snprintf(id, sizeof(id), "s%d", ++g_seq);
    Sess st; st.hw = s;
    g_sess[id] = st;
    g_last = id;
    return id;
}

std::vector<Msg> EngineHost::history(const std::string& sid) const {
    auto it = g_sess.find(sid);
    return it == g_sess.end() ? std::vector<Msg>{} : it->second.hist;
}

std::vector<SessionInfo> EngineHost::sessions() const {
    std::vector<SessionInfo> out;
    for (auto& [id, s] : g_sess) out.push_back({ id, s.title, s.hw ? s.hw->n_past : 0, s.turns });
    return out;
}

bool EngineHost::chat(const std::string& sid, const std::string& text) {
    auto it = g_sess.find(sid);
    if (it == g_sess.end() || !g_eng) return false;
    if (g_busy.exchange(true)) return false;
    g_cur_sid = sid;

    g_worker = std::jthread([this, sid, text] {
        auto& st = g_sess[sid];
        nova::Session* s = st.hw;
        auto& e = *g_eng;

        st.hist.push_back({ true, text });
        if (st.turns == 0) {
            std::string t = text;
            if (t.size() > 24) { t = t.substr(0, 24); t += "…"; }
            st.title = t;
        }

        // 增量 ChatML：首轮整段，之后只拼新增片段（历史 KV 已在池里）
        std::string delta;
        int64_t pos0 = 0;
        if (!s->kv_valid) {
            delta = nova::render_full(*s->card, s->history, s->sys);
            const std::string im_s = s->card->im_start, im_e = s->card->im_end;
            delta = im_s + "user\n" + text + im_e + "\n" + im_s + "assistant\n";
            pos0 = 0;
        } else {
            const std::string im_s = s->card->im_start, im_e = s->card->im_end;
            delta = im_e + "\n" + im_s + "user\n" + text + im_e + "\n" + im_s + "assistant\n";
            pos0 = s->n_past;
        }

        auto ids = nova::encode_chat(g_tk, delta);

        if (pos0 > 0) nova::session_cow(s, pos0, (int)ids.size());
        e.prefill(s->ws, ids.data(), (int)ids.size(), s->table_dev, pos0);
        s->n_past = pos0 + (int64_t)ids.size();
        s->kv_valid = true;

        const auto t0 = std::chrono::steady_clock::now();
        int cnt = 0;
        std::string answer;
        int tok = e.sample_last(s->ws);
        const int im_end = g_tk.id(s->card->im_end, 151645);
        for (int i = 0; i < 512 && tok != im_end; ++i) {
            std::string piece = g_tk.decode(std::vector<int>{ tok });
            answer += piece;
            { std::lock_guard<std::mutex> lk(g_mu); g_q.push(piece); }
            ++cnt;
            nova::session_cow(s, s->n_past, 1);
            tok = e.decode(s->ws, tok, s->table_dev, s->n_past);
            ++s->n_past;
        }

        st.hist.push_back({ false, answer });
        ++st.turns;

        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt > 0) g_tps = 0.6 * g_tps + 0.4 * (cnt / dt);
        g_busy = false;
    });
    return true;
}

void EngineHost::pump(const TokenFn& on_token) {
    for (;;) {
        std::string piece;
        { std::lock_guard<std::mutex> lk(g_mu); if (g_q.empty()) break; piece = std::move(g_q.front()); g_q.pop(); }
        on_token(piece);
    }
}

bool EngineHost::busy() const { return g_busy.load(); }

int64_t EngineHost::fork(const std::string& sid) {
    auto it = g_sess.find(sid);
    if (it == g_sess.end() || !it->second.hw) return -1;
    nova::Session* src = it->second.hw;
    const int64_t tokens = src->n_past;
    if (tokens <= 0) return -1;
    std::string nid = new_session();
    auto& dst = g_sess[nid];
    nova::session_attach_prefix(dst.hw, src, tokens);
    dst.hist = it->second.hist;
    dst.turns = it->second.turns;
    dst.title = it->second.title + " · fork";
    return (int64_t)tokens;
}

Telemetry EngineHost::telemetry(const std::string& sid) const {
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
    auto it = g_sess.find(sid.empty() ? g_last : sid);
    if (it != g_sess.end() && it->second.hw) t.n_past = it->second.hw->n_past;
    t.tok_s = g_tps;
    return t;
}