// code/main.h — 四路并发多智能体辩论（婚姻主题）
#pragma once
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "session.h"
#include "INPUT_CHAT.h"

namespace cfg {
inline constexpr const char* kModelPath = "build/q4k.gguf";
inline constexpr const char* kCardName  = "Qwen2.5-7B-Instruct";
inline constexpr int kCtx     = 2048;
inline constexpr int kTurn    = 32;
inline constexpr int kRounds  = 3;
inline constexpr const char* kTopic = "粽子到底该吃咸的还是甜的？请亮明立场。";
inline constexpr const char* kSys[4] = {
    "你是咸粽死忠。你坚信粽子必须是咸的：蛋黄、五花肉、酱油糯米才是正统。你语气激动，嘲笑甜粽是点心不是粽子。输出限制50字以内。",
    "你是甜粽死忠。你坚信粽子必须是甜的：豆沙、蜜枣、白糖才是灵魂。你觉得咸粽油腻难吃，语气不服气。输出限制50字以内。",
    "你是乐子人。你觉得咸甜都无聊，主张往粽子里加辣椒、加芝士、加榴莲。你就爱拱火看别人吵。输出限制50字以内。",
    "你根本不爱吃粽子。你觉得粽子就是一团碳水，咸甜之争毫无意义，你只想吃点别的。语气冷淡。输出限制50字以内。",
};
inline constexpr const char* kLabel[4] = {"A 咸粽党", "B 甜粽党", "C 辣椒邪教", "D 不爱吃"};
}

namespace nova {
inline const ModelCard* pick_card(const char* name) {
    for (int i = 0; i < kCardCount; ++i)
        if (std::strcmp(kCards[i].name, name) == 0) return &kCards[i];
    std::fprintf(stderr, "[ERROR] unsupported model: %s\n", name); std::exit(2);
}
inline void banner() {
    std::printf("\n");
    std::printf("   ______                 ____  _   __                 \n");
    std::printf("  / ____/___  ____  _____/ / / | / /___ _   ____ _     \n");
    std::printf(" / /   / __ \\/ __ \\/ ___/ / /  |/ / __ \\ | / / _ \\    \n");
    std::printf("/ /___/ /_/ / / / / /__/ / / /|  / /_/ / |/ /  __/    \n");
    std::printf("\\____/\\____/_/ /_/\\___/_/_/ /_|_/\\____/|___/\\___/     \n");
    std::printf("   >> V 2 . 0 . 0 - Core  |  Powered by 1080Ti <<\n\n");
}
} // namespace nova

int main(int argc, char** argv) {
    nova::banner();
    std::printf("[INFO] Conflux Nova V2.0.0  |  4-way debate\n");
    const char* path = argc > 1 ? argv[1] : cfg::kModelPath;

    nova::GgufFile meta(path);
    nova::Tokenizer tk; tk.load(meta);
    nova::Engine engine; engine.load(path);
    const nova::ModelCard* card = nova::pick_card(cfg::kCardName);
    nova::sessions().init(&engine);

    constexpr int N = 4;
    nova::Session* s[N];
    for (int i = 0; i < N; ++i) {
        s[i] = nova::sessions().create(card, &tk, cfg::kTurn);
        s[i]->sys = cfg::kSys[i];
    }
    std::printf("[SESSION] %d live, kv/session %.0f MB\n", nova::sessions().live_count(),
                (double)nova::Engine::kv_span() * 4 / 1048576.0);

    using clk = std::chrono::steady_clock;
    std::string prev[N];
    double total_ms = 0; int total_tok = 0;

    for (int round = 0; round < cfg::kRounds; ++round) {
        std::printf("\n########## ROUND %d ##########\n", round + 1);

        for (int i = 0; i < N; ++i) {
            std::string user;
            if (round == 0) user = (card->asst ? std::string(cfg::kSys[i]) + "\n\n" : std::string()) + cfg::kTopic;
            else {
                user = "刚才有人说：";
                int quoted = 0;
                for (int j = 0; j < N && quoted < 2; ++j) {
                    if (j == i) continue;
                    std::string q = prev[j];
                    if (q.size() > 36) q = q.substr(0, 36);   // 截断，别把上下文撑爆
                    user += "\"" + q + "\" ";
                    ++quoted;
                }
                user += "你怎么回？";
            }
            s[i]->history.push_back({"user", user});

            std::string delta; int64_t pos0 = 0;
            if (!s[i]->kv_valid) delta = nova::render_full(*card, s[i]->history, s[i]->sys);
            else {
                if (card->asst) delta = std::string(card->im_end) + card->im_start + user + card->asst;
                else delta = std::string(card->im_end) + "\n" + card->im_start + "user\n" + user + card->im_end + "\n" + card->im_start + "assistant\n";
                pos0 = s[i]->n_past;
            }
            auto ids = nova::encode_chat(tk, delta);
            engine.prefill(s[i]->ws, ids.data(), (int)ids.size(), s[i]->kvk, s[i]->kvv, pos0);
            s[i]->n_past = pos0 + (int64_t)ids.size();
            s[i]->kv_valid = true;
            s[i]->out_ids.clear();
            s[i]->out_ids.push_back(engine.sample_last(s[i]->ws));
        }

        const auto t0 = clk::now();
        const int im_end = tk.id(card->im_end, 128247);
        int last[N]; bool done[N];
        for (int i = 0; i < N; ++i) { last[i] = s[i]->out_ids[0]; done[i] = (last[i] == im_end); }
        nova::Engine::BatchItem bi[N];
        for (int t = 1; t < cfg::kTurn; ++t) {
            for (int i = 0; i < N; ++i) bi[i] = nova::Engine::BatchItem{last[i], s[i]->kvk, s[i]->kvv, s[i]->n_past, 0};
            engine.decode_batch(bi, N, s[0]->ws.s);
            for (int i = 0; i < N; ++i) {
                last[i] = bi[i].tok_out; ++s[i]->n_past;
                if (done[i]) continue;
                if (last[i] == im_end) { done[i] = true; continue; }
                s[i]->out_ids.push_back(last[i]);
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        total_ms += ms; total_tok += N * cfg::kTurn;

        for (int i = 0; i < N; ++i) {
            std::string txt = nova::clean_reply(tk.decode(s[i]->out_ids));
            prev[i] = txt;
            std::printf("[%s] %s\n", cfg::kLabel[i], txt.c_str());
            s[i]->history.push_back({"assistant", txt});
        }
        std::printf("[PERF] round %d: %d tok x%d in %.0f ms | %.1f tok/s per session | %.1f tok/s total\n",
                    round + 1, cfg::kTurn, N, ms, cfg::kTurn * 1000.0 / ms, cfg::kTurn * N * 1000.0 / ms);
        std::fflush(stdout);
    }

    std::printf("\n================ SUMMARY ================\n");
    std::printf("batch decode : %.0f ms total\n", total_ms);
    std::printf("aggregate    : %.2f tok/s\n", total_tok * 1000.0 / total_ms);
    std::printf("per session  : %.2f tok/s\n", cfg::kTurn * 1000.0 / (total_ms / cfg::kRounds));
    std::printf("kv lengths   : %lld / %lld / %lld / %lld\n", (long long)s[0]->n_past,
                (long long)s[1]->n_past, (long long)s[2]->n_past, (long long)s[3]->n_past);
    std::printf("live sessions: %d\n", nova::sessions().live_count());
    std::fflush(stdout);
    return 0;
}
