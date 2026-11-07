// tests/kv_demo.cpp — KV 块机制的最小验证：两轮对话，两个块，登记表可查
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
    if (!card) { std::printf("card not found\n"); return 1; }

    nova::Session* s = nova::sessions().create(card, &tk, 24);
    nova::blocks().set_capacity(8);

    nova::INPUT_CHAT(s, "用一句话说明什么是导数");
    nova::INPUT_CHAT(s, "那它的几何意义呢");

    std::printf("\n");
    nova::blocks().dump();

    const nova::KvBlock* b1 = nova::blocks().find(s->first_hash, nova::blocks().find(s->first_hash, 0) ? 0 : 0);
    (void)b1;
    return 0;
}
