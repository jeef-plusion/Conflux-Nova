// code/main.h — 总装：参数配置 + 控制台入口
#pragma once
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "kernel/1080ti/kernel_api.h"
#include "map/qwen2.5/model_map.h"
#include "map/qwen2.5/tokenizer.h"
#include "USER_INPUT_TXT.h"
#include "INPUT_INPUT.h"
#include "OUT_TKZ.h"
#include "OUT_chat.h"
#include "OUT_OUT.h"

namespace cfg {
// ---- 全部可调参数都在这里，内核不感知 ----
inline constexpr const char* kModelPath = "build/q4k.gguf";
inline constexpr const char* kCardName  = "Qwen2.5-7B-Instruct";
inline constexpr int  kCtx     = 2048;
inline constexpr int  kMaxNew  = 96;
inline constexpr const char* kQuestion = "请推导 f(x)=x^3-3x 的极值点";
}

namespace nova {

inline const ModelCard* pick_card(const char* name) {
    for (int i = 0; i < kCardCount; ++i)
        if (std::strcmp(kCards[i].name, name) == 0) return &kCards[i];
    std::fprintf(stderr, "[ERROR] 不支持的模型: %s\n", name);
    std::exit(2);
}

inline void banner() {
    std::printf("\n");
    std::printf("   ______                 ____  _   __                 \n");
    std::printf("  / ____/___  ____  _____/ / / | / /___ _   ____ _     \n");
    std::printf(" / /   / __ \\/ __ \\/ ___/ / /  |/ / __ \\ | / / _ \\    \n");
    std::printf("/ /___/ /_/ / / / / /__/ / / /|  / /_/ / |/ /  __/    \n");
    std::printf("\\____/\\____/_/ /_/\\___/_/_/ /_|_/\\____/|___/\\___/     \n");
    std::printf("                                                     \n");
    std::printf("   >> V 2 . 0 . 0 - Core  |  Powered by 1080Ti <<\n\n");
}

inline void print_models() {
    std::printf("[INFO] 支持的模型清单:\n");
    for (int i = 0; i < kCardCount; ++i)
        std::printf("       %-32s %-5s  ChatML\n", kCards[i].name, kCards[i].size);
}

inline int run(const char* path, const char* card_name, const char* question,
               int n_ctx, int max_new) {
    const ModelCard* card = pick_card(card_name);

    GgufFile meta(path);
    Tokenizer tk;
    tk.load(meta);

    Engine engine;
    engine.load(path, n_ctx);

    // 1) 文本 → 模板串
    const std::string prompt = build_prompt(*card, question);
    // 2) 模板串 → Token
    const std::vector<int> ids = encode_chat(tk, prompt);
    const int im_end = tk.id(card->im_end, 151645);

    std::printf("\n[USER] %s\n", question);
    std::printf("[TOKENS] prompt=%zu  first: ", ids.size());
    for (size_t i = 0; i < ids.size() && i < 12; ++i) std::printf("%d ", ids[i]);
    std::printf("...\n");
    std::fflush(stdout);

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    engine.prefill(ids.data(), (int)ids.size());
    const auto t1 = clk::now();

    std::vector<int> out;
    out.reserve(max_new);
    int next = ids.empty() ? im_end : ids.back();
    (void)next;
    // 首 token 由 prefill 后的 head 给出，直接从 decode 循环里取
    int tok = 0;
    {
        const std::vector<int> seed{ids.back()};
        (void)seed;
    }
    // 用最后一个 prompt token 走一次 decode，得到第一个回答 token
    // （prefill 已经把 KV 填好，这里相当于第 n 步）
    tok = engine.decode(ids.back());
    const auto t2 = clk::now();
    while ((int)out.size() < max_new) {
        if (tok == im_end) break;
        out.push_back(tok);
        tok = engine.decode(tok);
    }
    const auto t3 = clk::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const std::string text  = clean_reply(tokens_to_text(tk, out));
    emit(text, prefill_ms, decode_ms, (int)out.size());
    return 0;
}

} // namespace nova

int main(int argc, char** argv) {
    nova::banner();
    std::printf("[INFO] Conflux Nova V2.0.0\n");
    nova::print_models();
    const char* path = argc > 1 ? argv[1] : cfg::kModelPath;
    const char* q    = argc > 2 ? argv[2] : cfg::kQuestion;
    return nova::run(path, cfg::kCardName, q, cfg::kCtx, cfg::kMaxNew);
}
