// nova_v2/tests/sycl_scope_probe.cpp — 最小 SYCL 探针：profiling 队列 + 边界事件能不能用
//
// 为什么需要它：端到端带上 --scope 就崩（0xC0000409），而只开 NOVA_SCOPE=1（建 profiling 队列、
// 不记录）完全正常。所以嫌疑在"记录路径"：ext_oneapi_submit_barrier 与 get_profiling_info。
// 这个程序只做这一件事，崩在哪一行一目了然。
//
// 构建：icpx -fsycl -O2 -std=c++17 tests\sycl_scope_probe.cpp src\kernel\sycl\*.cpp ... 
//       （实际用 build_scope_probe.bat）
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

#include "model/sycl/scope_core.h"
#include "model/sycl/scope_sycl.h"

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("[probe] 1. 选择设备\n");
    sycl::device dev{sycl::default_selector_v};
    std::printf("[probe]    device = %s\n", dev.get_info<sycl::info::device::name>().c_str());

    std::printf("[probe] 2. 建 profiling 队列\n");
    sycl::queue q;
    try {
        const sycl::property_list props{sycl::property::queue::in_order{},
                                        sycl::property::queue::enable_profiling{}};
        q = sycl::queue(dev, props);
        std::printf("[probe]    ok\n");
    } catch (const std::exception& e) {
        std::printf("[probe]    FAIL: %s\n", e.what());
        return 1;
    }

    std::printf("[probe] 3. 三次发射 + 边界事件\n");
    std::vector<sycl::event> evs;
    constexpr int N = 3;
    for (int k = 0; k < N; ++k) {
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
            volatile float v = static_cast<float>(i[0]) * (k + 1);
            (void)v;
        });
        try {
            sycl::event ev = q.ext_oneapi_submit_barrier();
            evs.push_back(ev);
            std::printf("[probe]    第 %d 次 barrier ok\n", k);
        } catch (const std::exception& e) {
            std::printf("[probe]    第 %d 次 barrier FAIL: %s\n", k, e.what());
            return 2;
        }
    }
    q.wait();
    std::printf("[probe] 4. get_profiling_info\n");
    int okc = 0;
    for (std::size_t i = 0; i < evs.size(); ++i) {
        try {
            const auto s = evs[i].get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto e = evs[i].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::printf("[probe]    #%zu start=%llu end=%llu dur=%llu ns\n", i,
                        (unsigned long long)s, (unsigned long long)e,
                        (unsigned long long)(e - s));
            ++okc;
        } catch (const std::exception& ex) {
            std::printf("[probe]    #%zu FAIL: %s\n", i, ex.what());
        }
    }

    std::printf("[probe] 5. NovaScope 记录路径（Scope + op()）\n");
    {
        nova::scope::Scope::get().enable(true);
        nova::scope::Scope::get().set_token(0);
        for (int k = 0; k < N; ++k) {
            nova::scope::op("kernel", nova::scope::Phase::Compute, 1024, q, [&] {
                q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
                    volatile float v = static_cast<float>(i[0]);
                    (void)v;
                });
            });
            std::printf("[probe]    op %d 记录 ok\n", k);
        }
        std::printf("[probe]    resolve_all...\n");
        nova::scope::resolve_all(q);
        std::printf("[probe]    resolve ok，记录 %zu 条\n", nova::scope::Scope::get().recs().size());
        nova::scope::Scope::get().print_report(stdout);
        const bool w = nova::scope::Scope::get().write_trace("probe_trace.json");
        std::printf("[probe]    write_trace = %s\n", w ? "ok" : "FAILED");
    }

    std::printf("[probe] 结论：profiling=%d/3 可用，记录路径可用\n", okc);
    return 0;
}
