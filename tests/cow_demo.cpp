// tests/cow_demo.cpp — 跨会话共享 + COW 验证
#include <cstdio>
#include <cstring>
#include "session.h"
#include "INPUT_CHAT.h"
#include "graph/kv_registry.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "build/abl15b_q4k.gguf";
    nova::GgufFile meta(path);
    nova::Tokenizer tk; tk.load(meta);
    nova::Engine engine; engine.load(path);
    nova::sessions().init(&engine);

    const nova::ModelCard* card = nullptr;
    for (int i = 0; i < nova::kCardCount; ++i)
        if (std::strcmp(nova::kCards[i].name, "DeepSeek-R1-Distill-Qwen-1.5B-Abliterated") == 0) card = &nova::kCards[i];

    std::printf("[COW] pool: nb_total=%d nblocks=%d used=%d free_slices=%d\n",
                nova::kv_pool().nb_total(), nova::kv_pool().nblocks(),
                nova::kv_pool().used(), nova::kv_pool().free_slices());
    nova::Session* A = nova::sessions().create(card, &tk, 8);
    nova::Session* B = nova::sessions().create(card, &tk, 8);

    // A：一段前缀 + 4 个 token（块 0 只填了一部分）
    nova::INPUT_CHAT(A, "用一句话说明什么是导数");
    const int64_t shared = A->n_past;
    std::printf("[COW] A 的共享前缀长度 = %lld token（块 0 未填满）\n", (long long)shared);

    const int before = nova::kv_pool().cow_count();
    nova::session_attach_prefix(B, A, shared);
    std::printf("[COW] B 挂载后 n_past=%lld，refcnt(A块0)=%d\n",
                (long long)B->n_past, nova::kv_pool().refcnt(A->kslice->table[0]));

    // B 追加一个 token -> 落进被共享且半满的块 -> 应触发 COW
    const int lb = nova::kv_block_index(shared);          // 共享的最后一个逻辑块（半满）
    const int32_t old_phys = B->kslice->table[(size_t)lb];
    nova::session_cow(B, B->n_past, 1);
    const int32_t new_phys = B->kslice->table[(size_t)lb];
    const int after = nova::kv_pool().cow_count();

    std::printf("[COW] B 的块 %d: %d -> %d %s\n", lb, old_phys, new_phys, (old_phys != new_phys) ? "(已 COW)" : "(未变)");
    std::printf("[COW] cow_count: %d -> %d %s\n", before, after, (after > before) ? "(触发成功)" : "(未触发)");
    std::printf("[COW] A 的块 %d 仍是 %d，refcnt=%d（已减回 1）\n",
                lb, A->kslice->table[(size_t)lb], nova::kv_pool().refcnt(A->kslice->table[(size_t)lb]));
    std::printf("[COW] %s\n", (old_phys != new_phys && after > before) ? "COW PASS" : "COW FAIL");
    return 0;
}
