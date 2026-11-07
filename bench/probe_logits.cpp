// bench/probe_logits.cpp - 打印 prompt 之后 top-10 logits/token，用于判定前向是否健康
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include "nova/model.h"
#include "map/qwen2.5/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [text]\n", argv[0]); return 2; }
    const char* text = argc > 2 ? argv[2] : "Hello";
    nova::GgufFile g(argv[1]);
    nova::Tokenizer tk; tk.load(g);
    nova::Engine e; e.load(argv[1]);

    cudaStream_t s; cudaStreamCreate(&s);
    nova::Workspace ws; e.make_workspace(&ws, nova::kSessCtx, s);
    // 共享块池：整池清零，再用一张恒等块表（slice 0）
    int32_t* table = e.kv_table_slot(0);
    std::vector<int32_t> tab((size_t)nova::Engine::kv_nblocks());
    for (int b = 0; b < nova::Engine::kv_nblocks(); ++b) tab[(size_t)b] = b;
    cudaMemcpy(table, tab.data(), tab.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(e.kv_k_base(), 0, (size_t)nova::Engine::kv_nb() * nova::kKvBlock * nova::Engine::kv_width() * 4);
    cudaMemset(e.kv_v_base(), 0, (size_t)nova::Engine::kv_nb() * nova::kKvBlock * nova::Engine::kv_width() * 4);

    std::vector<int> ids;
    if (argc > 3 && std::string(argv[3]) == "chat") {
        ids.push_back(tk.id("<|im_start|>"));
        for (int v : tk.encode("user")) ids.push_back(v);
        for (int v : tk.encode("\n" + std::string(text))) ids.push_back(v);
        ids.push_back(tk.id("<|im_end|>"));
        for (int v : tk.encode("\n")) ids.push_back(v);
        ids.push_back(tk.id("<|im_start|>"));
        for (int v : tk.encode("assistant")) ids.push_back(v);
        for (int v : tk.encode("\n")) ids.push_back(v);
    } else {
        ids = tk.encode(text);
    }
    std::printf("[probe] text=\"%s\" tokens=%zu:\n        ", text, ids.size());
    for (int id : ids) std::printf("%d ", id);
    std::printf("\n");

    e.prefill(ws, ids.data(), (int)ids.size(), table, 0);
    std::vector<float> lg(nova::Model::V);
    cudaMemcpy(lg.data(), ws.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int> idx(nova::Model::V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    std::printf("[probe] top-10:\n");
    for (int k = 0; k < 10; ++k) {
        const int id = idx[k];
        std::string piece = tk.decode(std::vector<int>{id});
        for (auto& ch : piece) if (ch == '\n') ch = ' ';
        std::printf("   %2d. id=%-6d logit=%8.3f  \"%s\"\n", k + 1, id, lg[id], piece.c_str());
    }
    // 继续贪心 12 步，看 decode 路径是否健康
    std::printf("[probe] greedy continuation:\n   \"");
    int64_t pos = (int64_t)ids.size();
    int tok = e.sample_last(ws);
    for (int step = 0; step < 12; ++step) {
        std::string piece = tk.decode(std::vector<int>{tok});
        for (auto& ch : piece) if (ch == '\n') ch = ' ';
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        tok = e.decode(ws, tok, table, pos);
        ++pos;
    }
    std::printf("\"\n[probe] n_past=%lld\n", (long long)pos);
    return 0;
}
