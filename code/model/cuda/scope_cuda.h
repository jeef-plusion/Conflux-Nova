// nova_v2/code/model/cuda/scope_cuda.h — NovaScope 的 CUDA 探针
//
// 与 SYCL 版（code/model/sycl/scope_sycl.h）同一套统计核心（scope_core.h），只是把
// "取 device 时间"的手段换成 CUDA 事件：
//     cudaEventRecord(e0) -> kernel/memcpy -> cudaEventRecord(e1)
//     cudaEventElapsedTime(&ms, e0, e1)     // 结束时间差，单位 ms
//
// 三个关键设计：
//   1. **不在 begin/end 里同步**。老的 Profiler 每段都 cudaEventSynchronize，会打断流水线
//      并让 WDDM 的批处理失效 —— 那样量出来的"间隙"是探针自己造成的。这里只记录事件，
//      跑到最后统一 resolve_all() 一次同步。
//   2. **事件池复用**：一次 10 token ≈ 3400 条记录，逐条 create/destroy 事件本身就要几毫秒，
//      会污染测量。这里预分配 8192 对事件，用完即还。
//   3. 时间线用**累加**而不是绝对时刻：CUDA 事件只给"两事件之间的时长"，
//      所以按提交顺序把时长拼成一条时间轴（与 SYCL 版的边界差值同构）。
//
// 开关：环境变量 NOVA_SCOPE=1，或 Scope::get().enable(true)。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "model/sycl/scope_core.h"     // 复用纯 C++ 的 Rec / Summary / JSON / CSV

namespace nova {
namespace scope {

inline bool wanted() {
    static const bool env = [] {
        const char* e = std::getenv("NOVA_SCOPE");
        return e != nullptr && e[0] == '1';
    }();
    return env || Scope::get().enabled();
}

struct PendingC {
    std::size_t idx = 0;
    cudaEvent_t a = nullptr;
    cudaEvent_t b = nullptr;
};

class CudaEventPool {
public:
    ~CudaEventPool() {
        for (auto* e : free_) cudaEventDestroy(e);
        for (auto* e : used_) cudaEventDestroy(e);
    }
    cudaEvent_t take() {
        if (free_.empty()) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreateWithFlags(&e, cudaEventDefault) != cudaSuccess) return nullptr;
            used_.push_back(e);
            return e;
        }
        cudaEvent_t e = free_.back();
        free_.pop_back();
        return e;
    }
    void give(cudaEvent_t e) { if (e) free_.push_back(e); }

private:
    std::vector<cudaEvent_t> free_, used_;
};

inline CudaEventPool& pool() { static CudaEventPool p; return p; }
inline std::vector<PendingC>& pendings() { static std::vector<PendingC> v; return v; }

// 记录一次 launch 的 device 时长（不阻塞）
template <class F>
void op(const char* name, Phase ph, std::int64_t bytes, cudaStream_t s, F&& fn) {
    Scope& sc = Scope::get();
    if (!sc.enabled()) {
        fn();
        return;
    }
    const std::size_t idx = sc.recs().size();
    sc.begin(name, ph, bytes);

    cudaEvent_t a = pool().take();
    cudaEvent_t b = pool().take();
    if (!a || !b) {                 // 事件拿不到：退化成只有 host 时间
        fn();
        sc.finish(sc.recs()[idx]);
        pool().give(a);
        pool().give(b);
        return;
    }
    cudaEventRecord(a, s);
    fn();
    cudaEventRecord(b, s);
    pendings().push_back(PendingC{idx, a, b});
}

// 拷贝：同一个 op() 机制，只是 phase=Copy（H2D / D2H / D2D）
template <class F>
void copy_op(const char* name, std::int64_t bytes, cudaStream_t s, F&& fn) {
    op(name, Phase::Copy, bytes, s, [&] { fn(); });
}

// 一次同步，逐对算时长，再拼成设备时间线
inline void resolve_all(cudaStream_t s) {
    Scope& sc = Scope::get();
    if (!sc.enabled()) return;
    if (s) cudaStreamSynchronize(s);
    else   cudaDeviceSynchronize();

    std::int64_t prev = 0;
    for (auto& p : pendings()) {
        float ms = 0.f;
        const bool ok = (cudaEventElapsedTime(&ms, p.a, p.b) == cudaSuccess);
        Rec& r = sc.recs()[p.idx];
        if (!ok) {
            r.resolved = true;
            pool().give(p.a);
            pool().give(p.b);
            continue;
        }
        const std::int64_t d = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
        if (prev == 0) {
            r.start_ns = r.submit_ns;
            r.end_ns   = r.submit_ns + d;
        } else {
            r.start_ns = prev;
            r.end_ns   = prev + d;
        }
        r.resolved = true;
        // CUDA 事件给不出「host 提交 → device 开始」的延迟（那需要 CUPTI）。
        // 所以把报告时间轴上的提交时刻对齐到 start：gap 恒为 0（不编数），
        // 队列深度改由 Summary::host_span_ns / wall_ns 体现。
        r.submit_ns = r.start_ns;
        prev = r.end_ns;
        pool().give(p.a);
        pool().give(p.b);
    }
    pendings().clear();
}

// 打印 + 落盘（与 SYCL 版同一个报告器）
inline void finish_and_report(cudaStream_t s, const char* json_path, const char* csv_path) {
    Scope& sc = Scope::get();
    resolve_all(s);
    sc.print_report(stdout);
    const bool a = sc.write_trace(json_path);
    const bool b = csv_path ? sc.write_csv(csv_path) : true;
    std::printf("\n[scope] trace: %s (%s) | csv: %s (%s)\n", json_path, a ? "ok" : "FAILED",
                csv_path ? csv_path : "-", b ? "ok" : "FAILED");
    std::printf("[scope] 打开方式: chrome://tracing 或 ui.perfetto.dev 拖入 %s\n", json_path);
}

} // namespace scope
} // namespace nova

#define NOVA_SCOPED_CU(name, bytes, stream, ...) \
    ::nova::scope::op((name), ::nova::scope::Phase::Compute, (bytes), (stream), [&] { __VA_ARGS__; })

#define NOVA_SCOPED_COPY_CU(name, bytes, stream, ...) \
    ::nova::scope::copy_op((name), (bytes), (stream), [&] { __VA_ARGS__; })
