// tests/reuse_demo.cpp — 跨会话共享前缀：验证不污染、COW 生效、结果一致
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "session.h"
#include "INPUT_CHAT.h"
#include "graph/kv_registry.h"

static double block_sum(nova::Engine& e, int32_t phys, int layer) {
    const size_t per = (size_t)nova::kKvBlock * nova::Engine::kv_width();
    std::vector<float> h(per);
    cudaMemcpy(h.data(), e.kv_k_base() + ((size_t)layer * nova::Engine::kv_nb() + phys) * per,
               per * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0; for (float v : h) s += v; return s;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "build/abl15b_q4k.gguf";
    nova::GgufFile meta(path);
    nova::Tokenizer tk; tk.load(meta);
    nova::Engine engine; engine.load(path);
    nova::sessions().init(&engine);

    const nova::ModelCard* card = nullptr;
    for (int i = 0; i < nova::kCardCount; ++i)
        if (std::strcmp(nova::kCards[i].name, "DeepSeek-R1-Distill-Qwen-1.5B-Abliterated") == 0) card = &nova::kCards[i];

    // ---- A：共同前缀 ----
    nova::Session* A = nova::sessions().create(card, &tk, 16);
    std::string prompt = nova::render_full(*card, {{"user", "用一句话说明什么是导数"}}, "");
    auto ids = nova::encode_chat(tk, prompt);
    engine.prefill(A->ws, ids.data(), (int)ids.size(), A->table_dev, 0);
    A->n_past = (int64_t)ids.size();
    const int64_t P = A->n_past;

    // A 贪心生成 8 个 token
    std::vector<int> outA;
    int tok = engine.sample_last(A->ws);
    std::vector<int> first = {tok};
    for (int i = 0; i < 8; ++i) {
        outA.push_back(tok);
        nova::session_cow(A, A->n_past, 1);
        tok = engine.decode(A->ws, tok, A->table_dev, A->n_past);
        ++A->n_past;
    }

    // 记录 A 共享块的内容指纹
    const int lb = nova::kv_block_index(P);
    const int32_t physA = A->kslice->table[(size_t)lb];
    const double sum_before = block_sum(engine, physA, 0);
    std::printf("[REUSE] A: prefix=%lld tok, 共享块 lb=%d phys=%d sum=%.6f\n",
                (long long)P, lb, physA, sum_before);

    // ---- B：挂载同一前缀 ----
    nova::Session* B = nova::sessions().create(card, &tk, 16);
    const int cow0 = nova::kv_pool().cow_count();
    nova::session_attach_prefix(B, A, P);

    // 挂载后 B 没有 logits：用最后一个 prompt token 在原地重算一次（写入同样的值）
    nova::session_cow(B, P - 1, 1);
    std::vector<int> outB;
    int tb = engine.decode(B->ws, ids.back(), B->table_dev, P - 1);
    for (int i = 0; i < 8; ++i) {
        outB.push_back(tb);
        nova::session_cow(B, B->n_past, 1);
        tb = engine.decode(B->ws, tb, B->table_dev, B->n_past);
        ++B->n_past;
    }
    const int cow1 = nova::kv_pool().cow_count();

    const double sum_after = block_sum(engine, physA, 0);
    std::printf("[REUSE] B: 生成 %zu tok, COW %d 次\n", outB.size(), cow1 - cow0);
    std::printf("[REUSE] A 共享块指纹: %.6f -> %.6f  %s\n", sum_before, sum_after,
                (std::fabs(sum_before - sum_after) < 1e-6) ? "(未被污染)" : "(被改写！)");
    std::printf("[REUSE] 文本一致性: %s\n", (outA == outB) ? "A/B 输出完全相同" : "A/B 输出不同");

    std::printf("[REUSE] A: %s\n", tk.decode(outA).c_str());
    std::printf("[REUSE] B: %s\n", tk.decode(outB).c_str());

    const bool ok = (std::fabs(sum_before - sum_after) < 1e-6) && (cow1 > cow0) && (outA == outB);
    std::printf("[REUSE] %s\n", ok ? "REUSE PASS" : "REUSE FAIL");
    return ok ? 0 : 1;
}
