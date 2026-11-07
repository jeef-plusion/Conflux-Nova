// headless_main.cpp — 不开窗口，验证三条链路：选卡 / 流式 / Fork+COW
#include "engine_host.h"
#include <cstdio>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    EngineHost host;
    std::string err;
    const char* path = argc > 1 ? argv[1] : "build/q4k.gguf";

    auto t0 = std::chrono::steady_clock::now();
    if (!host.load(path, err)) { std::printf("[FAIL] load: %s\n", err.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("[OK] load %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    // ---- 流式验证 ----
    std::printf("[CHAT] 请推导 f(x)=x^3-3x 的极值点\n[STREAM] ");
    host.chat("请推导 f(x)=x^3-3x 的极值点");
    int n = 0; const auto ts = std::chrono::steady_clock::now();
    while (host.busy() || n == 0) {
        host.pump([&](const std::string& tok) { std::printf("%s", tok.c_str()); std::fflush(stdout); ++n; });
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (n > 0 && !host.busy()) break;
    }
    host.pump([&](const std::string& tok) { std::printf("%s", tok.c_str()); ++n; });
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
    std::printf("\n[STREAM] %d tok, %.2f s, %.1f tok/s\n", n, dt, n / (dt > 0 ? dt : 1));

    auto tA = host.telemetry();
    std::printf("[TELE] n_past=%lld vram=%.2f/%.2f blocks=%d/%d shared=%d cow=%d tps=%.1f\n",
                (long long)tA.n_past, tA.vram_used_gb, tA.vram_cap_gb,
                tA.blocks_total - tA.blocks_free, tA.blocks_total, tA.blocks_shared, tA.cow_count, tA.tok_s);

    // ---- Fork 验证 ----
    const auto& tr = host.tree();
    if (tr.size() > 1) {
        const int64_t blk = tr.back().block;
        const int64_t f = host.fork(blk);
        auto tB = host.telemetry();
        std::printf("[FORK] block=%lld -> %lld | shared %d -> %d, cow=%d\n",
                    (long long)blk, (long long)f, tA.blocks_shared, tB.blocks_shared, tB.cow_count);
    }
    return 0;
}