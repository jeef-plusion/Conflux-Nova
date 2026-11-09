// nova_v2/tests/sycl_e2e.cpp — SYCL 端到端 decode：真模型、真 28 层、真采样
//
// 用法:
//   sycl_e2e.exe --gguf build\q4k_15b.gguf [--prompt "你好"] [--tokens 32]
//                [--ctx 512] [--sample] [--warmup 2]
//
// 做的事：
//   1. 从 GGUF 读 tokenizer（词表 + merges），把 ChatML 提示词编码成 id；
//   2. 逐 token 喂进 runner（== CUDA 版 Engine::decode 的路径：embed/28 层/head/采样）；
//   3. 打印每 token 的 ms、decode tok/s、以及解出来的文本。
//
// 退出码 = 0（跑通）/ 1（参数或加载失败）。
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "map/qwen2.5/tokenizer.h"
#include "nova/loader/gguf.h"
#include "runner.h"

namespace {

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Chat 模板：不同家族的回合标记不一样，而且**模板必须能整体变成一个 token**，
// 否则模型收到的是字面文本（这正是第一版输出的乱码原因）。
//   ChatML（Qwen2.5 / R1-distill）：<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
//   R1-Abliterated 变种：<｜User｜>...<｜end▁of▁sentence｜><｜Assistant｜>   （全角 ｜ = U+FF5C）
enum class Tpl { Auto, ChatML, R1, None };

const char* tpl_name(Tpl t) {
    switch (t) {
        case Tpl::ChatML: return "chatml";
        case Tpl::R1:     return "r1-abliterated";
        case Tpl::None:   return "none";
        default:          return "auto";
    }
}

Tpl detect_tpl(const nova::Tokenizer& tok) {
    if (tok.id("<|im_start|>") >= 0) return Tpl::ChatML;
    if (tok.id("<" "\xef\xbd\x9c" "User" "\xef\xbd\x9c" ">") >= 0) return Tpl::R1;
    return Tpl::None;
}

std::string build_prompt(Tpl t, const std::string& user) {
    switch (t) {
        case Tpl::ChatML:
            return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        case Tpl::R1:
            return "<" "\xef\xbd\x9c" "User" "\xef\xbd\x9c" ">" + user +
                   "<" "\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence" "\xef\xbd\x9c" ">" +
                   "<" "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">";
        default:
            return user;
    }
}

void usage() {
    std::printf(
        "用法: sycl_e2e.exe --gguf <path> [选项]\n"
        "  --gguf   <path>   Q4_K GGUF（默认 build/q4k_15b.gguf）\n"
        "  --prompt <text>   用户消息（默认 \"你好\"）\n"
        "  --tokens <n>      生成多少个 token（默认 32）\n"
        "  --ctx    <n>      KV 逻辑长度（默认 512）\n"
        "  --warmup <n>      生成前的预热次数（默认 2）\n"
        "  --template <t>    chatml | r1 | none | auto（默认 auto：按词表探测）\n"
        "  --sample          走 top-k/top-p/temp 采样（默认 greedy，可复现）\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string gguf   = "build/q4k_15b.gguf";
    std::string prompt = "你好";
    int  n_gen    = 32;
    int  ctx      = 512;
    int  warmup   = 2;
    bool greedy   = true;
    Tpl  tpl      = Tpl::Auto;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::printf("[e2e] %s 缺少参数\n", what); return nullptr; }
            return argv[++i];
        };
        if (s == "--gguf")      { const char* v = next("--gguf");   if (!v) return 1; gguf = v; }
        else if (s == "--prompt")   { const char* v = next("--prompt"); if (!v) return 1; prompt = v; }
        else if (s == "--tokens")   { const char* v = next("--tokens"); if (!v) return 1; n_gen = std::atoi(v); }
        else if (s == "--ctx")      { const char* v = next("--ctx");    if (!v) return 1; ctx = std::atoi(v); }
        else if (s == "--warmup")   { const char* v = next("--warmup"); if (!v) return 1; warmup = std::atoi(v); }
        else if (s == "--template") {
            const char* v = next("--template"); if (!v) return 1;
            const std::string t = v;
            if (t == "chatml") tpl = Tpl::ChatML;
            else if (t == "r1") tpl = Tpl::R1;
            else if (t == "none") tpl = Tpl::None;
            else tpl = Tpl::Auto;
        }
        else if (s == "--sample")   greedy = false;
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { std::printf("[e2e] 未知参数: %s\n", s.c_str()); usage(); return 1; }
    }
    if (n_gen < 1) n_gen = 1;
    if (n_gen + 128 > ctx) { std::printf("[e2e] --tokens 太大（KV 只有 %d 个位置）\n", ctx); return 1; }

    std::printf("=== SYCL 端到端 decode（model=%s）===\n", kSyclModelName);

    // ---- 分词器 ----
    nova::GgufFile g(gguf);
    nova::Tokenizer tok;
    tok.load(g);
    std::printf("[e2e] 模型: %s（%s 个 token）\n", nova::general_name(g).c_str(),
                std::to_string(g.str_array("tokenizer.ggml.tokens").size()).c_str());

    if (tpl == Tpl::Auto) tpl = detect_tpl(tok);
    if (tpl == Tpl::None && prompt.empty()) {
        std::printf("[e2e] FATAL: 词表里既没有 <|im_start|> 也没有 <｜User｜>，且没给 --prompt\n");
        return 1;
    }
    const std::string text = build_prompt(tpl, prompt);
    std::printf("[e2e] 模板: %s\n", tpl_name(tpl));

    const std::vector<int> ids = tok.encode(text);
    std::printf("[e2e] prompt %zu 个 token: ", ids.size());
    for (size_t i = 0; i < ids.size() && i < 16; ++i) std::printf("%d ", ids[i]);
    if (ids.size() > 16) std::printf("...");
    std::printf("\n");

    // ---- runner ----
    nova::sycls::E2ERunner runner(gguf, ctx);
    std::printf("[e2e] 权重 %.1f MiB，ctx=%d\n",
                static_cast<double>(runner.weight_bytes()) / (1024.0 * 1024.0), runner.ctx_max());

    if (warmup > 0) {
        std::printf("[e2e] 预热 %d 次...\n", warmup);
        runner.warmup(warmup);
    }

    // ---- 逐 token 喂 prompt（位置 0..n-1；最后一个 token 的输出就是第一个生成 token）----
    const auto t_pre0 = Clock::now();
    int next = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        next = runner.step(ids[i], static_cast<int>(i), greedy);
    }
    const double t_pre = secs(t_pre0, Clock::now());
    std::printf("[e2e] prompt 处理 %zu 个 token，%.2f ms（%.1f tok/s）\n",
                ids.size(), t_pre * 1e3, ids.size() / t_pre);

    // ---- 生成 ----
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n_gen));
    double t_gen = 0.0;
    for (int i = 0; i < n_gen; ++i) {
        out.push_back(next);
        if (i + 1 == n_gen) break;                       // 最后一个不需要再前进一步
        const auto t0 = Clock::now();
        next = runner.step(next, static_cast<int>(ids.size()) + i, greedy);
        t_gen += secs(t0, Clock::now());
    }

    const double per_tok = t_gen / std::max(1, n_gen - 1);
    std::printf("\n[e2e] 生成 %d 个 token：%.2f ms/token -> %.2f tok/s（decode，不含 prompt）\n",
                n_gen, per_tok * 1e3, 1.0 / per_tok);
    std::printf("[e2e] 采样: %s\n", greedy ? "greedy" : "top-k/top-p/temp");

    // ---- 文本 ----
    const std::string gen_text = tok.decode(out);
    std::printf("\n--- 生成 token id ---\n");
    for (int id : out) std::printf("%d ", id);
    std::printf("\n\n--- 生成文本 ---\n%s\n", gen_text.c_str());

    return 0;
}
