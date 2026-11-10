// nova_v2/tests/cuda_scope.cpp — NovaScope 的 CUDA 端到端驱动（1080 Ti）
//
// 做的事很小，但每件事都必须是真的：
//   1. 真加载 GGUF（Q4_K），真跑 28/32 层，真采样；
//   2. 每个算子由 Profiler 记录一对 cudaEvent（不阻塞），拷自由 NOVA_SCOPED_COPY_CU 记账；
//   3. 跑完统一 resolve_all()，打印四组统计，写出 nova_trace_cuda.json / .csv。
//
// 用法:
//   set NOVA_SCOPE=1
//   cuda_scope.exe <model.gguf> [生成 token 数]
//
// 为什么单独写驱动而不是改 code/main.cpp：main 是"四路辩论"演示，参数和流程都不适合定量；
// 这里用 Engine 的最小 API（load / make_workspace / prefill / decode），一次 decode 就是
// 一个 token 的完整前向，和 SYCL 版 runner 的口径一致。

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "nova/model.h"
#include "model/cuda/scope_cuda.h"

namespace {

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count() * 1e3;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* path = (argc > 1) ? argv[1] : "build/q4k.gguf";
    const int   n_gen = (argc > 2) ? std::atoi(argv[2]) : 16;

    std::printf("=== NovaScope / CUDA 端到端（%s，%d token）===\n", path, n_gen);
    nova::scope::Scope::get().enable(nova::scope::wanted());
    std::printf("[scope] %s\n", nova::scope::Scope::get().enabled() ? "已开启（事件对 + 非阻塞）" : "未开启");

    nova::Engine eng;
    eng.load(path);
    std::printf("[nova] 权重 %.1f MiB\n", static_cast<double>(eng.weight_bytes()) / (1024.0 * 1024.0));

    // profiler 走 scoped 路径：不每段同步
    eng.prof.scoped = nova::scope::Scope::get().enabled();

    nova::Workspace w{};
    eng.make_workspace(&w, 1, nullptr);      // 1 列，默认流

    // 块表：非共享会话用恒等映射（逻辑块 i -> 物理块 i）
    const int nb = nova::Engine::kv_nblocks();
    std::vector<std::int32_t> h_table(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) h_table[static_cast<std::size_t>(i)] = i;
    std::int32_t* table = nullptr;
    cudaMalloc(&table, sizeof(std::int32_t) * static_cast<std::size_t>(nb));
    cudaMemcpy(table, h_table.data(), sizeof(std::int32_t) * static_cast<std::size_t>(nb),
               cudaMemcpyHostToDevice);

    // 一段短 prompt（id 只要在词表内即可，这里用 ASCII 的 "Hello"）
    const int prompt[] = {9707, 11, 1879, 0};
    int pos = 0;
    int next = -1;
    for (int id : prompt) next = eng.decode(w, id, table, pos++);
    std::printf("[nova] prompt %d token 走完，pos=%d\n", 4, pos);

    if (nova::scope::Scope::get().enabled()) {
        nova::scope::resolve_all(w.s);                    // prompt 阶段也收一遍（暖机数据）
        nova::scope::Scope::get().clear();
        nova::scope::pendings().clear();
    }

    // ---- 正式测：连续 decode ----
    const auto t0 = Clock::now();
    for (int i = 0; i < n_gen; ++i) {
        nova::scope::Scope::get().set_token(i);
        next = eng.decode(w, next, table, pos++);
    }
    const double wall_ms = ms_between(t0, Clock::now());
    std::printf("[nova] 生成 %d token: %.2f ms/token -> %.2f tok/s\n", n_gen, wall_ms / n_gen,
                n_gen / (wall_ms / 1e3));

    if (nova::scope::Scope::get().enabled()) {
        std::printf("\n[nova] 采样基准：%.2f ms/token（%.2f tok/s，含采样与 D2H）\n",
                    wall_ms / n_gen, n_gen / (wall_ms / 1e3));
        nova::scope::finish_and_report(w.s, "nova_trace_cuda.json", "nova_trace_cuda.csv");
    }

    cudaFree(table);
    return 0;
}
