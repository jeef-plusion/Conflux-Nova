// nova_v2/tests/sycl_soa_bench.cpp — AoS vs SoA 的纯带宽对照（不碰生产内核）
//
// 要回答的问题：Q4_K 每个 super-block 是 144 B（16 B 块头 + 128 B qs）。现在 AoS 布局下
// 8 个 lane 读 qs 是完美合并的，但块头那 16 B 落在独立的 64 B sector 上 —— 每块占
// 3 个 sector = 192 B，理论上限 61.38 × 144/192 = 46 GB/s。SoA（块头集中存放）后
// 一行里 4 个块头正好占 1 个 sector，每块回到 144 B，上限 61.38。
//
// 这个基准只做"读"，把两种访问模式各自的真实带宽量出来，避免先花大力气重构、
// 结果发现 sector 模型不成立。
//
// 用法: sycl_soa_bench.exe [MiB] [reps]
// 构建: icpx -fsycl -O2 -std=c++17 -Iinclude -Icode tests\sycl_soa_bench.cpp

#include <cstdio>
#include <cstdlib>

#include <vector>

#include <sycl/sycl.hpp>

namespace {

constexpr int kQK = 256;          // 一个 super-block 的权重数
constexpr int kQsBytes = 128;     // 128 B qs
constexpr int kHdrBytes = 16;     // 16 B 块头（d/dmin/scales）
constexpr int kBlkBytes = kQsBytes + kHdrBytes;   // 144 B
constexpr int kLanes = 8;         // 每条 lane 读 16 B qs

// AoS：和现在生产内核一样的读法 —— lane 读 qs，所有 lane 读同一个块头
void read_aos(const std::uint8_t* w, int nblk, int lane, std::int64_t* sink) {
    std::int64_t acc = 0;
    for (int b = 0; b < nblk; ++b) {
        const std::uint8_t* blk = w + static_cast<std::int64_t>(b) * kBlkBytes;
        // 8 个 lane 各读 16 B qs（合并成 128 B）
        const sycl::vec<std::int32_t, 4> q =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(blk + kHdrBytes + lane * 16);
        // 每个 lane 都读同一个 16 B 块头（硬件广播，但仍占 1 个 sector）
        const sycl::vec<std::int32_t, 4> h =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(blk);
        acc += q.x() + q.y() + q.z() + q.w() + h.x() + h.w();
    }
    *sink += acc;
}

// SoA：qs 与 hdr 各自连续；lane 读 qs，一个 lane 组一次把 4 个块头（64 B）读进来
void read_soa(const std::uint8_t* qs, const std::uint8_t* hdr, int nblk, int lane,
              std::int64_t* sink) {
    std::int64_t acc = 0;
    for (int b = 0; b < nblk; b += 4) {
        // 4 个块头 = 64 B = 恰好一个 sector（所有 lane 读同一组，广播）
        const sycl::vec<std::int32_t, 4> h0 =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(hdr + static_cast<std::int64_t>(b) * kHdrBytes);
        const sycl::vec<std::int32_t, 4> h1 =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(hdr + static_cast<std::int64_t>(b + 1) * kHdrBytes);
        const sycl::vec<std::int32_t, 4> h2 =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(hdr + static_cast<std::int64_t>(b + 2) * kHdrBytes);
        const sycl::vec<std::int32_t, 4> h3 =
            *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(hdr + static_cast<std::int64_t>(b + 3) * kHdrBytes);
        acc += h0.x() + h1.x() + h2.x() + h3.x();
        // 4 个块的 qs：每块 8 lane × 16 B
        for (int t = 0; t < 4; ++t) {
            const sycl::vec<std::int32_t, 4> q = *reinterpret_cast<const sycl::vec<std::int32_t, 4>*>(
                qs + static_cast<std::int64_t>(b + t) * kQsBytes + lane * 16);
            acc += q.x() + q.y() + q.z() + q.w();
        }
    }
    *sink += acc;
}

// 网格和生产内核同构：每组 8 lane，负责一段独立的 block 区间（这样才有足够并行度喂满 DRAM）
double bench(const char* tag, sycl::queue& q, int kind, const std::uint8_t* a, const std::uint8_t* b,
             int nblk, int reps, int lanes) {
    std::int64_t* sink = sycl::malloc_shared<std::int64_t>(1, q);
    constexpr int kGroupBlocks = 38;                       // 每组处理 38 个块（近似 down 的 BPR=35）
    const int ngrp = nblk / kGroupBlocks;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < reps; ++it) {
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(ngrp) * kLanes, kLanes),
                                         sycl::range<2>(kLanes, kLanes)),
                       [=](sycl::nd_item<2> it2) {
            const int g = static_cast<int>(it2.get_group(0));
            const int lane = static_cast<int>(it2.get_local_id(1));
            const int b0 = g * kGroupBlocks;
            if (kind == 0) read_aos(a + static_cast<std::int64_t>(b0) * kBlkBytes, kGroupBlocks, lane, sink);
            else           read_soa(a + static_cast<std::int64_t>(b0) * kQsBytes,
                                    b + static_cast<std::int64_t>(b0) * kHdrBytes,
                                    kGroupBlocks, lane, sink);
        });
    }
    q.wait();
    const double ms =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    sycl::free(sink, q);

    const double bytes = static_cast<double>(nblk) * kBlkBytes * reps;
    const double gbs = bytes / (ms * 1e-3) / 1e9;
    std::printf("  %-28s %8.2f ms  %8.1f MiB  %7.2f GB/s\n", tag, ms, bytes / 1048576.0, gbs);
    return gbs;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int mib = (argc > 1) ? std::atoi(argv[1]) : 8;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 200;

    sycl::device dev{sycl::default_selector_v};
    sycl::queue q{dev};
    std::printf("=== AoS vs SoA 纯读带宽（Arc）===\n");
    std::printf("device: %s | 每次读 %d MiB（必须大于 L2，才是真 DRAM）\n",
                dev.get_info<sycl::info::device::name>().c_str(), mib);

    const int nblk = (mib * 1048576) / kBlkBytes;
    std::printf("blocks = %d（每块 144 B）\n\n", nblk);

    std::uint8_t* aos = sycl::malloc_device<std::uint8_t>(static_cast<std::size_t>(nblk) * kBlkBytes, q);
    std::uint8_t* qs  = sycl::malloc_device<std::uint8_t>(static_cast<std::size_t>(nblk) * kQsBytes, q);
    std::uint8_t* hdr = sycl::malloc_device<std::uint8_t>(static_cast<std::size_t>(nblk) * kHdrBytes, q);
    q.memset(aos, 0x33, static_cast<std::size_t>(nblk) * kBlkBytes);
    q.memset(qs, 0x33, static_cast<std::size_t>(nblk) * kQsBytes);
    q.memset(hdr, 0x33, static_cast<std::size_t>(nblk) * kHdrBytes);
    q.wait();

    // 预热
    bench("warmup", q, 0, aos, nullptr, nblk, 10, 1024);
    bench("warmup", q, 1, qs, hdr, nblk, 10, 1024);

    const double g_aos = bench("AoS (144 B/block)", q, 0, aos, nullptr, nblk, reps, 1024);
    const double g_soa = bench("SoA (128 B + 16 B/4blk)", q, 1, qs, hdr, nblk, reps, 1024);

    std::printf("\n  AoS 相对上限：61.38 × 144/192 = 46.0 GB/s -> 实测 %.1f%%\n", 100.0 * g_aos / 46.0);
    std::printf("  SoA 相对上限：61.38            -> 实测 %.1f%%\n", 100.0 * g_soa / 61.38);
    std::printf("  提升：%.2fx（%.1f -> %.1f GB/s）\n", g_soa / g_aos, g_aos, g_soa);

    sycl::free(aos, q); sycl::free(qs, q); sycl::free(hdr, q);
    return 0;
}
