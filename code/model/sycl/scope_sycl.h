// nova_v2/code/model/sycl/scope_sycl.h — NovaScope 的 SYCL 事件探针
//
// 核心思路（为什么用「边界事件差值」而不是直接取 kernel 事件）：
//   我们的算子层是 `void gemv(...)` 这类签名 —— 它们内部 `q.parallel_for(...)` 把 event 丢掉了。
//   要拿每个 kernel 的 device start/end，有两条路：
//     (a) 把所有算子改成返回 sycl::event（侵入 15 个函数、15 个声明）；
//     (b) 在每次发射后提交一个**队列边界**，用相邻边界的结束时间差当作该 op 的占用时间。
//   这里选 (b)：队列是 in_order，语义上"前一个 op 跑完才跑下一个"，所以
//     op_i 的 device 区间 = [barrier_{i-1}.end, barrier_i.end]
//   与 (a) 逐 kernel 取值的差别只有"边界自身的开销"（实测 ~1~2 µs），换来零侵入。
//   如果以后要 exact per-kernel 数值，再把 (a) 作为 `NOVA_SCOPE_EXACT` 补上。
//
// 前置条件：队列必须带 `property::queue::enable_profiling`（device.cpp 按 NOVA_SCOPE=1 建）。
// 环境变量 NOVA_SCOPE=1 打开；不打开时 op() 直接转发，不取时间、不分配。
#pragma once

#if defined(__SYCL_COMPILER_VERSION) || defined(__INTEL_LLVM_COMPILER) || defined(NOVA_SCOPE_SYCL)

#include <cstdlib>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "model/sycl/scope_core.h"

namespace nova {
namespace scope {

// 开关来源有两个：环境变量 NOVA_SCOPE=1，或命令行 --scope 先把 Scope 打开。
inline bool wanted() {
    static const bool env = [] {
        const char* e = std::getenv("NOVA_SCOPE");
        return e != nullptr && e[0] == '1';
    }();
    return env || Scope::get().enabled();
}

struct Pending {
    std::size_t  idx = 0;
    sycl::event  ev;
};

inline std::vector<Pending>& pendings() {
    static std::vector<Pending> v;
    return v;
}

// 包裹一次发射：fn() 提交，随后立刻提交一个边界事件
template <class F>
void op(const char* name, Phase ph, std::int64_t bytes, sycl::queue& q, F&& fn) {
    Scope& s = Scope::get();
    if (!s.enabled()) {
        fn();
        return;
    }
    const std::size_t idx = s.recs().size();
    s.begin(name, ph, bytes);
    fn();
    sycl::event ev = q.ext_oneapi_submit_barrier();
    pendings().push_back(Pending{idx, ev});
}

// 拷贝：把 memcpy 返回的 event 交进来（H2D / D2H / D2D 都用它）
inline void note_copy(sycl::event ev, const char* name, std::int64_t bytes) {
    Scope& s = Scope::get();
    if (!s.enabled()) return;
    const std::size_t idx = s.recs().size();
    s.copy(name, bytes);
    pendings().push_back(Pending{idx, ev});
}

inline void note_copy(sycl::queue& q, const char* name, std::int64_t bytes) {
    // 没有 event 可用时（例如 UMA 上的隐式迁移）也记一条，时长记 0，靠 host 时间兜底
    (void)q;
    Scope& s = Scope::get();
    if (!s.enabled()) return;
    s.copy(name, bytes);
}

// 排空：q.wait() 之后把每个边界事件的结束时间写回 Rec，再换算成区间
inline void resolve_all(sycl::queue& q) {
    Scope& s = Scope::get();
    if (!s.enabled()) return;
    q.wait();

    std::vector<std::int64_t> ends;
    ends.reserve(pendings().size());
    std::vector<std::size_t> idxs;
    idxs.reserve(pendings().size());

    for (auto& p : pendings()) {
        std::int64_t e = 0;
        try {
            e = static_cast<std::int64_t>(
                p.ev.get_profiling_info<sycl::info::event_profiling::command_end>());
        } catch (...) {
            e = 0;   // 队列没开 profiling：保持 host 时间，不编数
        }
        ends.push_back(e);
        idxs.push_back(p.idx);
    }

    std::int64_t prev_end = 0;
    for (std::size_t k = 0; k < ends.size(); ++k) {
        Rec& r = s.recs()[idxs[k]];
        const std::int64_t e = ends[k];
        if (e == 0) {                 // 拿不到事件时间 -> 保留 host 侧时间
            r.resolved = true;
            prev_end = 0;
            continue;
        }
        if (prev_end == 0) {
            r.start_ns = r.submit_ns; // 第一个边界：从提交算起
            r.end_ns   = e;
        } else {
            r.start_ns = prev_end;    // 串行队列：上一个边界结束就是本 op 开始
            r.end_ns   = e;
        }
        r.resolved = true;
        prev_end = e;
    }
    pendings().clear();
}

} // namespace scope
} // namespace nova

#define NOVA_SCOPED(name, bytes, ...)                                                        \
    ::nova::scope::op((name), ::nova::scope::Phase::Compute, (bytes), q_, [&] { __VA_ARGS__; })

#define NOVA_SCOPED_COPY(name, bytes, ex)     \
    ::nova::scope::note_copy((ex), (name), (bytes))

#else   // 没有 SYCL：MSVC 下也能 include（变成空宏），方便排查

#include "model/sycl/scope_core.h"

#define NOVA_SCOPED(name, bytes, ...) __VA_ARGS__
#define NOVA_SCOPED_COPY(name, bytes, ex) (ex)

#endif
