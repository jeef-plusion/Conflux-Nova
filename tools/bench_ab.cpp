// bench_ab.cpp - 同一进程内 A/B 交替 + 热漂移检测（消除笔记本降频噪声）
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double sdev(const std::vector<double>& v, double m) {
    double s = 0; for (double x : v) s += (x - m) * (x - m);
    return v.size() > 1 ? std::sqrt(s / (v.size() - 1)) : 0.0;
}

// 只读扫描：每字节读 1 次，测纯带宽
static double scan_gbps(sycl::queue& q, const std::uint8_t* p, std::size_t bytes, float* sink) {
    const std::size_t n = bytes / 16;   // 16B = uint4 粒度
    auto t0 = clk::now();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) {
            const auto i = it.get_id(0);
            const sycl::vec<std::uint8_t, 16> v = *reinterpret_cast<const sycl::vec<std::uint8_t, 16>*>(p + i * 16);
            float s = 0.f;
            for (int k = 0; k < 16; ++k) s += (float)v[k];
            if (s == 1.2345e30f) sink[0] = s;   // 防止被优化掉
        });
    });
    q.wait();
    auto t1 = clk::now();
    return (double)bytes / (ms(t0, t1) * 1e6);   // GB/s
}

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        auto dev = q.get_device();
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

        const std::size_t MB = 512;
        const std::size_t bytes = MB * 1024ull * 1024ull;
        auto* d = (std::uint8_t*)sycl::malloc_device(bytes, q);
        auto* s = (std::uint8_t*)sycl::malloc_shared(bytes, q);
        float* sink = sycl::malloc_shared<float>(1, q);
        if (!d || !s) { std::printf("alloc failed\n"); return 1; }
        q.memset(d, 1, bytes).wait();
        q.memset(s, 1, bytes).wait();

        // 预热
        scan_gbps(q, d, bytes, sink);
        scan_gbps(q, s, bytes, sink);

        const int ROUNDS = 12;
        std::vector<double> A, B;
        std::printf("\n%4s %10s %10s\n", "轮次", "device", "shared");
        for (int i = 0; i < ROUNDS; ++i) {
            const double a = scan_gbps(q, d, bytes, sink);
            const double b = scan_gbps(q, s, bytes, sink);
            if (i > 0) { A.push_back(a); B.push_back(b); }   // 剔除第一次（冷启动）
            std::printf("%4d %9.2f %10.2f\n", i, a, b);
        }

        const double ma = med(A), mb = med(B);
        std::printf("\n中位数 : device %.2f GB/s | shared %.2f GB/s\n", ma, mb);
        std::printf("标准差 : device %.2f | shared %.2f\n", sdev(A, ma), sdev(B, mb));

        // 漂移：前 1/3 与后 1/3 的中位数之比
        const size_t k = A.size() / 3;
        std::vector<double> a1(A.begin(), A.begin() + k), a3(A.end() - k, A.end());
        std::vector<double> b1(B.begin(), B.begin() + k), b3(B.end() - k, B.end());
        std::printf("漂移   : device %.2f -> %.2f (%+.1f%%) | shared %.2f -> %.2f (%+.1f%%)\n",
                    med(a1), med(a3), 100.0 * (med(a3) - med(a1)) / med(a1),
                    med(b1), med(b3), 100.0 * (med(b3) - med(b1)) / med(b1));
        std::printf("结论   : %s 更快，差距 %.1f%%（标准差 %.1f%%，%s）\n",
                    ma > mb ? "device" : "shared", 100.0 * std::abs(ma - mb) / std::min(ma, mb),
                    100.0 * sdev(A, ma) / ma,
                    std::abs(ma - mb) / std::max(sdev(A, ma), 1e-9) > 2 ? "差异显著" : "在噪声内");

        sycl::free(d, q); sycl::free(s, q); sycl::free(sink, q);
    } catch (const std::exception& e) { std::printf("FATAL %s\n", e.what()); return 1; }
    return 0;
}