// tests/cow_e2e_demo.cpp — 物理共享、逻辑隔离的端到端验收
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "session.h"
#include "INPUT_CHAT.h"
#include "graph/kv_registry.h"

static std::string sys_prefix(const nova::ModelCard& c) {
    std::string p;
    p += c.im_start; p += "system\n"; p += "你是一个精通数学和物理的严谨助手，请用精简的语言回答问题。"; p += c.im_end; p += "\n";
    return p;
}
static std::string user_turn(const nova::ModelCard& c, const char* q) {
    std::string p;
    p += c.im_start; p += "user\n"; p += q; p += c.im_end; p += "\n";
    p += c.im_start; p += "assistant\n";
    return p;
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

    nova::Session* A = nova::sessions().create(card, &tk, 32);
    nova::Session* B = nova::sessions().create(card, &tk, 32);

    // 1) A 载入 system 前缀
    const auto sys_ids = nova::encode_chat(tk, sys_prefix(*card));
    int64_t Ps = (int64_t)sys_ids.size();
    // 让前缀不要正好卡在块边界上，否则尾块是满的、永远不会触发 COW
    while (Ps > 1 && (Ps % nova::kKvBlock) == 0) --Ps;
    engine.prefill(A->ws, sys_ids.data(), (int)Ps, A->table_dev, 0);
    A->n_past = Ps;
    std::printf("[E2E] system 前缀 = %lld token，跨越 %d 个块\n",
                (long long)Ps, (int)nova::kv_block_index(Ps - 1) + 1);

    // 2) A 的问题 + 生成
    const auto qa = nova::encode_chat(tk, user_turn(*card, "什么是导数？"));
    engine.prefill(A->ws, qa.data(), (int)qa.size(), A->table_dev, Ps);
    A->n_past = Ps + (int64_t)qa.size();
    std::vector<int> outA;
    int ta = engine.sample_last(A->ws);
    for (int i = 0; i < 10; ++i) {
        outA.push_back(ta);
        ta = engine.decode(A->ws, ta, A->table_dev, A->n_past);
        ++A->n_past;
    }

    // 3) B 挂载 system 前缀（只共享 [0, Ps)）
    const int cow0 = nova::kv_pool().cow_count();
    nova::session_attach_prefix(B, A, Ps);
    const int32_t shared_tail = A->kslice->table[(size_t)nova::kv_block_index(Ps - 1)];
    std::printf("[E2E] B 挂载 %lld token；共享尾块 phys=%d refcnt=%d\n",
                (long long)Ps, shared_tail, nova::kv_pool().refcnt(shared_tail));

    // B 的 prefill：第一笔写入落在共享尾块 -> 应触发 COW
    const auto qb = nova::encode_chat(tk, user_turn(*card, "什么是引力？"));
    nova::session_cow(B, Ps, (int)qb.size());
    engine.prefill(B->ws, qb.data(), (int)qb.size(), B->table_dev, Ps);
    B->n_past = Ps + (int64_t)qb.size();
    const int cow_after_prefill = nova::kv_pool().cow_count();

    std::vector<int> outB;
    int tb = engine.sample_last(B->ws);
    std::vector<int> seq;
    for (int i = 0; i < 10; ++i) {
        outB.push_back(tb);
        nova::session_cow(B, B->n_past, 1);
        tb = engine.decode(B->ws, tb, B->table_dev, B->n_past);
        ++B->n_past;
        seq.push_back(nova::kv_pool().cow_count());
    }

    const std::string tA = tk.decode(outA), tB = tk.decode(outB);
    std::printf("[E2E] A: %s\n", tA.c_str());
    std::printf("[E2E] B: %s\n", tB.c_str());

    bool cow_once = (cow_after_prefill - cow0) == 1;
    bool no_more = true; for (int v : seq) if (v != cow_after_prefill) no_more = false;
    std::printf("[E2E] COW: prefill 触发 %d 次%s；之后 10 步序列 ", cow_after_prefill - cow0, cow_once ? "（正确）" : "（异常）");
    for (int v : seq) std::printf("%d ", v);
    std::printf("%s\n", no_more ? "（未再触发，正确）" : "（又触发了，异常）");

    const int32_t now_tail = A->kslice->table[(size_t)nova::kv_block_index(Ps - 1)];
    std::printf("[E2E] A 的共享尾块 phys=%d refcnt=%d（B COW 后应回到 1）\n",
                now_tail, nova::kv_pool().refcnt(now_tail));
    std::printf("[E2E] A 自己的后续块 refcnt 采样: ");
    for (int b = (int)nova::kv_block_index(Ps); b < (int)nova::kv_block_index(A->n_past); ++b)
        std::printf("%d ", nova::kv_pool().refcnt(A->kslice->table[(size_t)b]));
    std::printf("\n");

    const bool a_ok = tA.find("引力") == std::string::npos && tA.find("导") != std::string::npos;
    const bool b_ok = tB.find("导数") == std::string::npos && (tB.find("引") != std::string::npos || tB.find("力") != std::string::npos);
    std::printf("[E2E] 语义隔离: A %s | B %s\n", a_ok ? "只谈导数" : "异常", b_ok ? "只谈引力" : "异常");

    const bool ok = cow_once && no_more && a_ok && b_ok && nova::kv_pool().refcnt(now_tail) == 1;
    std::printf("[E2E] %s\n", ok ? "E2E PASS" : "E2E FAIL");
    return ok ? 0 : 1;
}
