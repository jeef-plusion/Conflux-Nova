// bench/benchmark.cpp — 2048 上下文 prefill + 128 token decode，含 CUDA Event 算子分布
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "nova/model.h"

using namespace nova;

static double now_ms() {
    cudaDeviceSynchronize();
    static cudaEvent_t a = nullptr, b = nullptr;
    if (!a) { cudaEventCreate(&a); cudaEventCreate(&b); }
    cudaEventRecord(a);
    cudaEventSynchronize(a);
    return 0.0;
}

static size_t used_mb(size_t total) {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return (total - f) / (1024 * 1024);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [ctx=2048] [ngen=128]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const int ctx  = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int ngen = argc > 3 ? std::atoi(argv[3]) : 128;

    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    std::printf("[bench] GPU %.0f MB, 初始空闲 %.0f MB\n", total / 1048576.0, free0 / 1048576.0);

    Engine e;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    e.load(path, ctx);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float load_ms = 0;
    cudaEventElapsedTime(&load_ms, t0, t1);
    const size_t used_after_load = used_mb(total);
    std::printf("[bench] 加载+上传 %.0f ms, 显存占用 %zu MB\n", load_ms, used_after_load);

    std::vector<int> ids(ctx);
    for (int i = 0; i < ctx; ++i) ids[i] = 9707 + (i % 97);

    // 预热 3 次
    for (int w = 0; w < 3; ++w) {
        e.reset();
        e.prefill(ids.data(), ctx);
        for (int i = 0; i < 8; ++i) e.decode(ids[i]);
    }
    std::printf("[bench] 预热完成 3 轮\n");

    // prefill 计时
    double best_prefill = 1e30;
    for (int r = 0; r < 3; ++r) {
        e.reset();
        cudaEventRecord(t0);
        e.prefill(ids.data(), ctx);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        const double tps = ctx * 1000.0 / ms;
        std::printf("[bench] prefill %d tok: %.2f ms  %.1f tok/s\n", ctx, ms, tps);
        if (ms < best_prefill) best_prefill = ms;
    }
    std::printf("[bench] prefill 最佳: %.2f ms  %.1f tok/s\n", best_prefill, ctx * 1000.0 / best_prefill);

    // decode 计时
    e.reset();
    e.prefill(ids.data(), ctx);
    std::vector<int> out;
    out.reserve(ngen);
    cudaEventRecord(t0);
    for (int i = 0; i < ngen; ++i) out.push_back(e.decode(out.empty() ? ids[0] : out.back()));
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float dms = 0;
    cudaEventElapsedTime(&dms, t0, t1);
    std::printf("[bench] decode %d tok: %.2f ms  %.2f tok/s (%.3f ms/tok)\n",
                ngen, dms, ngen * 1000.0 / dms, dms / ngen);
    std::printf("[bench] 输出 token ids:");
    for (int i = 0; i < ngen && i < 16; ++i) std::printf(" %d", out[i]);
    std::printf(" ...\n");

    const size_t peak = used_mb(total);
    std::printf("[bench] 显存峰值(采样): %zu MB\n", peak);

    // 算子耗时分布
    e.prof.on = true;
    for (int i = 0; i < PH_COUNT; ++i) e.prof.acc[i] = 0;
    e.reset();
    e.prefill(ids.data(), ctx);
    e.prof.on = false;
    double sum = 0;
    for (int i = 0; i < PH_COUNT; ++i) sum += e.prof.acc[i];
    std::printf("[prof] prefill %d tok 总计 %.2f ms\n", ctx, sum);
    const char* names[PH_COUNT] = {"embed", "norm", "gemv/matmul", "attention", "rope+kv", "elementwise", "head"};
    for (int i = 0; i < PH_COUNT; ++i)
        std::printf("[prof]   %-12s %8.2f ms  %5.1f%%\n", names[i], e.prof.acc[i], 100.0 * e.prof.acc[i] / sum);

    e.prof.on = true;
    for (int i = 0; i < PH_COUNT; ++i) e.prof.acc[i] = 0;
    int tok = ids[0];
    for (int i = 0; i < 32; ++i) tok = e.decode(tok);
    e.prof.on = false;
    sum = 0;
    for (int i = 0; i < PH_COUNT; ++i) sum += e.prof.acc[i];
    std::printf("[prof] decode 32 tok 总计 %.2f ms (%.3f ms/tok)\n", sum, sum / 32);
    for (int i = 0; i < PH_COUNT; ++i)
        std::printf("[prof]   %-12s %8.2f ms  %5.1f%%\n", names[i], e.prof.acc[i], 100.0 * e.prof.acc[i] / sum);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
