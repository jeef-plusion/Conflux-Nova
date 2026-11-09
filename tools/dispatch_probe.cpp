// dispatch_probe.cpp - 空 kernel 的提交/执行开销（回答：dispatch 是不是瓶颈）
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template <class Q>
static void run(const char* tag, Q& q) {
    const int N = 2000;
    float* buf = sycl::malloc_shared<float>(64, q);
    for (int i = 0; i < 20; ++i)
        q.parallel_for(sycl::range<1>(64), [=](sycl::item<1> it) { buf[it.get_id(0)] = 1.f; });
    q.wait();

    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        q.parallel_for(sycl::range<1>(64), [=](sycl::item<1> it) { buf[it.get_id(0)] = 2.f; });
    auto t1 = clk::now();
    q.wait();
    auto t2 = clk::now();
    std::printf("%-14s submit-only : %8.2f us/dispatch\n", tag, ms(t0, t1) * 1000.0 / N);
    std::printf("%-14s submit+drain: %8.2f us/dispatch\n", tag, ms(t0, t2) * 1000.0 / N);

    auto t3 = clk::now();
    for (int i = 0; i < N; ++i) {
        q.parallel_for(sycl::range<1>(64), [=](sycl::item<1> it) { buf[it.get_id(0)] = 3.f; });
        q.wait();
    }
    auto t4 = clk::now();
    std::printf("%-14s submit+wait  : %8.2f us/dispatch\n", tag, ms(t3, t4) * 1000.0 / N);

    auto t5 = clk::now();
    q.parallel_for(sycl::range<1>(64 * N), [=](sycl::item<1> it) { buf[it.get_id(0) % 64] = 4.f; });
    q.wait();
    auto t6 = clk::now();
    std::printf("%-14s 1 big kernel   : %8.2f us total\n", tag, ms(t5, t6) * 1000.0);
    sycl::free(buf, q);
}

int main() {
    try {
        sycl::queue qo{sycl::gpu_selector_v};
        std::printf("device: %s\n\n", qo.get_device().get_info<sycl::info::device::name>().c_str());
        run("out-of-order", qo);
        std::printf("\n");
        sycl::queue qi{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        run("in-order", qi);
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}