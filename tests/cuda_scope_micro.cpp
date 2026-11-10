// nova_v2/tests/cuda_scope_micro.cpp — 用真实 Q4_K GEMV 内核验证 NovaScope 的 CUDA 探针
//
// 为什么要有这个：1080 Ti 这台机器现在有**进程级显存配额**（实测第 4 次分配必失败，
// 即便报 8.3 GiB 空闲），整模型加载不了。但探针本身对不对，不需要整模型就能验：
// 拿真实 kernel + 真实 cudaEvent 对，把 100 次 GEMV 的 device 时间与带宽量出来，
// 生成的 trace 与仪表盘路径和整模型完全一致。
//
// 用法: cuda_scope_micro.exe [行数] [列数] [重复次数]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "nova/kernel.h"
#include "model/cuda/scope_cuda.h"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // 用引擎现成的 Kernel::gate_n：形状固定 D x FFN = 1536 x 8960（Q4_K 7.37 MiB），
    // 模板实例已存在，带宽也够大（超过 1080 Ti 的 2.75 MiB L2，能测到真 DRAM）。
    constexpr int rows = 8960;
    constexpr int cols = 1536;
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 200;

    std::printf("=== NovaScope / CUDA 微基准（真实 Q4_K GEMV 内核）===\n");
    std::printf("形状 %d x %d，重复 %d 次，权重 %.1f MiB\n", rows, cols, reps,
                (double)rows * (cols / 256) * 144 / (1024.0 * 1024.0));

    nova::scope::Scope::get().enable(true);

    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);

    // 权重：只要字节数对，内容不影响带宽测量
    std::uint8_t* wblk = nullptr;
    const std::size_t wbytes = (std::size_t)rows * (cols / 256) * 144;
    cudaMalloc(&wblk, wbytes);
    cudaMemset(wblk, 0x22, wbytes);

    // 激活（Q8）：qs / d / s 三段
    std::int8_t* xq = nullptr;
    float* xd = nullptr;
    int* xs = nullptr;
    cudaMalloc(&xq, (std::size_t)cols);
    cudaMalloc(&xd, (std::size_t)(cols / 32) * 4);
    cudaMalloc(&xs, (std::size_t)(cols / 32) * 4);
    cudaMemset(xq, 1, (std::size_t)cols);
    cudaMemset(xd, 0, (std::size_t)(cols / 32) * 4);
    cudaMemset(xs, 0, (std::size_t)(cols / 32) * 4);

    float* out = nullptr;
    cudaMalloc(&out, (std::size_t)rows * 4);
    cudaMemset(out, 0, (std::size_t)rows * 4);

    nova::Q4KWeight w{};
    w.blocks = wblk;
    w.rows = rows;
    w.cols = cols;
    nova::ActQ8 x{};
    x.qs = xq;
    x.d = xd;
    x.s = xs;

    // 预热（不进 trace）
    for (int i = 0; i < 5; ++i) nova::Kernel::gate_n(w, x, out, 1, s);
    cudaStreamSynchronize(s);

    // ---- 正式测：每次 launch 一对事件 ----
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        nova::scope::Scope::get().set_token(0);
        NOVA_SCOPED_CU("gemv_q4k", (std::int64_t)wbytes, s, nova::Kernel::gate_n(w, x, out, 1, s));
    }
    const double wall_ms =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;

    // 注意：这个墙钟只量到"host 提交完"，GPU 还在异步跑（所以它看起来比设备时间短很多）。
    // 真实带宽看下面报告里的「权重流量 / GPU 忙」——那是 device 事件算出来的。
    std::printf("\n[nova] host 提交 %d 次用了 %.2f ms（异步，不代表 GPU 已跑完）\n", reps, wall_ms);

    nova::scope::finish_and_report(s, "nova_trace_cuda.json", "nova_trace_cuda.csv");

    cudaFree(wblk); cudaFree(xq); cudaFree(xd); cudaFree(xs); cudaFree(out);
    cudaStreamDestroy(s);
    return 0;
}
