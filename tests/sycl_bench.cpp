// nova_v2/tests/sycl_bench.cpp — SYCL 后端基准 + 数值对拍
//
// 目标机上两条命令出数据：
//     build_sycl.bat
//     build\sycl\sycl_bench.exe --gguf build\q4k.gguf
//
// 报告（缺一不可）：
//   1. 从 GGUF 里读真实张量 blk.0.attn_q.weight（Q4_K 3584x3584），跑 GEMV 100 次；
//   2. 与 CPU 参考实现的 rel_l2（要求 < 1e-4）；
//   3. 有效带宽 GB/s（= 权重字节 / 平均耗时）；
//   4. dp4a 加速比（同一 kernel 模板参数 HW=false 再跑一次）；
//   5. 按 1.5B / 7B 推算的 decode tok/s。
// 另外附：gemv4 四列吞吐、冷缓存（绕开 L3）带宽、其余算子的数值抽检。
//
// 退出码 = 未通过的性能门槛数（0 = 全过）。详见 docs/SYCL_PORT.md §6。
#include <sycl/sycl.hpp>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nova/config.h"
#include "nova/loader/gguf.h"
#include "nova/model/matrix.h"
#include "nova/quant/q4_k.h"

#include "device.h"
#include "kernels.h"

namespace syclb = nova::syclb;
using nova::syclb::ActQ8View;
using nova::syclb::ChunkedWeight;
using nova::syclb::Device;
using nova::syclb::MemTag;

namespace {

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
struct Args {
    std::string gguf   = "build/q4k.gguf";
    std::string tensor = "blk.0.attn_q.weight";
    int  iters  = 100;
    int  warmup = 5;
    int  ctx    = 512;
    bool cold   = true;
    bool checks = true;
};

void usage() {
    std::printf(
        "用法: sycl_bench.exe [选项]\n"
        "  --gguf   <path>   默认 build/q4k.gguf（7B 的 Q4_K 文件）\n"
        "  --tensor <name>   默认 blk.0.attn_q.weight\n"
        "  --iters  <n>      计时迭代数（默认 100）\n"
        "  --warmup <n>      预热迭代数（默认 5）\n"
        "  --ctx    <n>      推算 tok/s 用的上下文长度（默认 512）\n"
        "  --no-cold         跳过冷缓存（绕 L3）测量\n"
        "  --no-checks       跳过其它算子的数值抽检\n");
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::printf("[bench] %s 缺少参数\n", what); return nullptr; }
            return argv[++i];
        };
        if (s == "--gguf")        { const char* v = next("--gguf"); if (!v) return false; a.gguf = v; }
        else if (s == "--tensor") { const char* v = next("--tensor"); if (!v) return false; a.tensor = v; }
        else if (s == "--iters")  { const char* v = next("--iters"); if (!v) return false; a.iters = std::atoi(v); }
        else if (s == "--warmup") { const char* v = next("--warmup"); if (!v) return false; a.warmup = std::atoi(v); }
        else if (s == "--ctx")    { const char* v = next("--ctx"); if (!v) return false; a.ctx = std::atoi(v); }
        else if (s == "--no-cold")   a.cold = false;
        else if (s == "--no-checks") a.checks = false;
        else if (s == "-h" || s == "--help") { usage(); return false; }
        else { std::printf("[bench] 未知参数: %s\n", s.c_str()); usage(); return false; }
    }
    if (a.iters < 2) a.iters = 2;
    return true;
}

// ---------------------------------------------------------------------------
// 计时：N 次发射 + 一次 wait，返回每次平均秒数。
// 差分法（N 与 2N）消掉 host 提交开销，得到"纯 kernel"时间。
// ---------------------------------------------------------------------------
template <typename F>
double avg_launch(F&& f, int n, sycl::queue& q) {
    const auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) f();
    q.wait_and_throw();
    const auto t1 = Clock::now();
    return secs(t0, t1) / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// ActQ8 缓冲
// ---------------------------------------------------------------------------
struct ActBuf {
    std::int8_t*  qs = nullptr;
    float*        d  = nullptr;
    std::int32_t* s  = nullptr;
    int           rows = 0;
    ActQ8View view() const { return ActQ8View{qs, d, s}; }
};

ActBuf alloc_act(Device& dev, int rows, int K, const char* what) {
    ActBuf b;
    b.rows = rows;
    b.qs = static_cast<std::int8_t*>(dev.alloc(syb::act_q8_bytes(rows, K), MemTag::Act, what));
    b.d  = static_cast<float*>(dev.alloc(syb::act_d_bytes(rows, K), MemTag::Act, what));
    b.s  = static_cast<std::int32_t*>(dev.alloc(syb::act_s_bytes(rows, K), MemTag::Act, what));
    return b;
}

// ---------------------------------------------------------------------------
// 参考实现
// ---------------------------------------------------------------------------
// out[r] = sum_i dequant(w[r,i]) * d[i/32] * qs[i]   （double 累加，独立于 kernel 的代数重排）
void cpu_gemv_ref(const std::uint8_t* w, int rows, int cols,
                  const std::int8_t* qs, const float* d, double* out) {
    std::vector<float> row(static_cast<std::size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        nova::dequant_row(w, r, cols, row.data());
        double acc = 0.0;
        for (int i = 0; i < cols; ++i) {
            acc += static_cast<double>(row[static_cast<std::size_t>(i)]) *
                   static_cast<double>(d[i >> 5]) *
                   static_cast<double>(qs[i]);
        }
        out[r] = acc;
    }
}

struct ErrStat {
    double rel_l2 = 0.0;
    double max_abs = 0.0;
};

ErrStat compare(const float* got, const double* ref, int n) {
    ErrStat e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(got[i]) - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
        const double ad = std::fabs(diff);
        if (ad > e.max_abs) e.max_abs = ad;
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

ErrStat compare_f32(const float* got, const float* ref, int n) {
    ErrStat e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += diff * diff;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        const double ad = std::fabs(diff);
        if (ad > e.max_abs) e.max_abs = ad;
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

// 确定性伪随机（不依赖 <random> 的实现差异）
struct Rng {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const std::uint32_t u = static_cast<std::uint32_t>(s >> 32);
        return (static_cast<float>(u) / 2147483648.0f) - 1.0f;   // [-1, 1)
    }
};

void section(const char* title) {
    std::printf("\n=== %s ===\n", title);
}
void rule() { std::printf("--------------------------------------------------------------\n"); }

} // namespace

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    std::printf("nova_v2 SYCL 后端基准 / model=%s (K=%d, NH=%d, NKV=%d, HD=%d, FFN=%d, V=%d, L=%d)\n",
                nova::syclk::kSyclModelName, nova::syclk::kD, nova::syclk::kNH, nova::syclk::kNKV,
                nova::syclk::kHD, nova::syclk::kFFN, nova::syclk::kV, nova::syclk::kL);
    syb::Device::print_banner();

    // -----------------------------------------------------------------------
    // 0. 设备
    // -----------------------------------------------------------------------
    section("0. 设备");
    Device dev;
    std::printf("%s", dev.caps().report().c_str());
    if (!syb::kDp4aHardware) {
        std::printf("\n[WARN] 本次构建的 dp4a 路径 = %s\n"
                    "       这不是硬件 dp4a。build_sycl.bat 的探针没找到 sycl::dp4a 的两个候选写法，\n"
                    "       第 4 项（dp4a 加速比）必然 ≈ 1.0。见 docs/SYCL_PORT.md §7.2。\n",
                    syb::dp4a_mode_name(syb::kNovaDp4aMode));
    }

    // -----------------------------------------------------------------------
    // 1. 读真实张量
    // -----------------------------------------------------------------------
    section("1. 从 GGUF 读张量");
    std::printf("[bench] file   = %s\n", args.gguf.c_str());
    nova::GgufFile g(args.gguf);
    const nova::TensorInfo* ti = g.find(args.tensor);
    if (ti == nullptr) {
        std::printf("[bench] FATAL: 找不到张量 %s。用 tools/list_tensors 核对名字。\n", args.tensor.c_str());
        return 2;
    }
    if (ti->type != nova::GgmlType::Q4_K) {
        std::printf("[bench] FATAL: 张量类型不是 Q4_K（该文件可能是 Q6_K 或其它）。\n");
        return 2;
    }
    if (ti->ne.size() != 2) {
        std::printf("[bench] FATAL: 期望二维张量，实际 %zu 维。\n", ti->ne.size());
        return 2;
    }
    const int cols = static_cast<int>(ti->ne[0]);
    const int rows = static_cast<int>(ti->ne[1]);
    std::printf("[bench] tensor = %s  type=Q4_K  cols(K)=%d  rows=%d  bytes=%" PRIu64 "\n",
                args.tensor.c_str(), cols, rows, ti->bytes);
    if (cols != nova::syclk::kD) {
        std::printf("[bench] FATAL: K=%d 与本二进制的 kD=%d 不一致；请用对应的 -DNOVA_SYCL_MODEL_* 重编。\n",
                    cols, nova::syclk::kD);
        return 2;
    }
    if (ti->bytes != static_cast<std::uint64_t>(rows) * syb::q4k_row_bytes(cols)) {
        std::printf("[bench] FATAL: 字节数与 rows*row_bytes 不符。\n");
        return 2;
    }

    std::vector<std::uint8_t> host_w(static_cast<std::size_t>(ti->bytes));
    {
        std::ifstream f(args.gguf, std::ios::binary);
        if (!f) { std::printf("[bench] FATAL: 打不开文件。\n"); return 2; }
        f.seekg(static_cast<std::streamoff>(g.data_offset() + ti->offset), std::ios::beg);
        f.read(reinterpret_cast<char*>(host_w.data()), static_cast<std::streamsize>(host_w.size()));
        if (!f) { std::printf("[bench] FATAL: 读张量数据失败。\n"); return 2; }
    }
    {
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < host_w.size(); i += 4096) sum += host_w[i];
        std::printf("[bench] 前 16 B:");
        for (int i = 0; i < 16; ++i) std::printf(" %02x", host_w[static_cast<std::size_t>(i)]);
        std::printf("   （每 4096 B 抽样求和 = 0x%016" PRIx64 "，用于确认读到的是真数据）\n", sum);
    }

    // -----------------------------------------------------------------------
    // 2. 权重上卡（分块）+ 激活
    // -----------------------------------------------------------------------
    section("2. 权重 / 激活");
    ChunkedWeight w;
    w.create(dev, rows, syb::q4k_row_bytes(cols), "bench.attn_q");
    w.upload(dev, host_w.data());
    std::printf("[bench] 权重分块: %d 块, 共 %.2f MiB（每块 ≤ %.0f MiB）\n", w.n_chunks(),
                static_cast<double>(w.bytes()) / (1024.0 * 1024.0),
                static_cast<double>(syb::profile::kMaxChunkBytes) / (1024.0 * 1024.0));

    constexpr int K = nova::syclk::kD;
    const int nb_d = K / 32;
    const int nb_s = K / 16;

    ActBuf act = alloc_act(dev, 1, K, "bench.act");
    std::vector<float> host_x(static_cast<std::size_t>(K));
    {
        Rng rng;
        for (int i = 0; i < K; ++i) host_x[static_cast<std::size_t>(i)] = rng.next() * 2.5f;
    }
    // fp32 激活 -> 设备 -> 量化（走的就是引擎里那条 quant_rows）
    float* dx = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(K) * 4, MemTag::Io, "bench.x"));
    dev.upload(dx, host_x.data(), static_cast<std::uint64_t>(K) * 4);
    nova::syclk::quant_rows<K>(act.view(), dx, 1, dev.q());
    dev.sync();

    std::vector<std::int8_t> h_qs(static_cast<std::size_t>(K));
    std::vector<float>       h_d(static_cast<std::size_t>(nb_d));
    std::vector<std::int32_t> h_s(static_cast<std::size_t>(nb_s));
    dev.download(h_qs.data(), act.qs, static_cast<std::uint64_t>(K));
    dev.download(h_d.data(), act.d, static_cast<std::uint64_t>(nb_d) * 4);
    dev.download(h_s.data(), act.s, static_cast<std::uint64_t>(nb_s) * 4);
    std::printf("[bench] 激活: %d 个 int8, %d 个 scale, %d 个 16-元素和\n", K, nb_d, nb_s);

    // -----------------------------------------------------------------------
    // 3. CPU 参考 + 正确性
    // -----------------------------------------------------------------------
    section("3. GEMV 与 CPU 参考对拍");
    int check_fail = 0;   // 数值抽检的失败计数（门槛汇总里一起算）
    float* d_out = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(rows) * 4, MemTag::Io, "bench.out"));
    std::vector<double> h_ref(static_cast<std::size_t>(rows));
    const auto t_ref0 = Clock::now();
    cpu_gemv_ref(host_w.data(), rows, cols, h_qs.data(), h_d.data(), h_ref.data());
    const double t_ref = secs(t_ref0, Clock::now());
    std::printf("[bench] CPU 参考完成：%d 行 x %d 列，double 累加，耗时 %.1f ms\n", rows, cols, t_ref * 1e3);

    nova::syclk::gemv<K, true>(w, act.view(), d_out, dev.q());
    dev.sync();
    std::vector<float> h_got(static_cast<std::size_t>(rows));
    dev.download(h_got.data(), d_out, static_cast<std::uint64_t>(rows) * 4);
    const ErrStat err = compare(h_got.data(), h_ref.data(), rows);
    std::printf("[bench] rel_l2 = %.3e   max_abs = %.3e   门槛 < 1e-4 -> %s\n",
                err.rel_l2, err.max_abs, (err.rel_l2 < 1e-4) ? "PASS" : "FAIL");
    {
        std::printf("[bench] 抽样 5 行: gpu / ref\n");
        for (int r : {0, 1, rows / 2, rows - 2, rows - 1}) {
            std::printf("        row %-5d  %+12.6f  %+12.6f\n", r,
                        static_cast<double>(h_got[static_cast<std::size_t>(r)]), h_ref[static_cast<std::size_t>(r)]);
        }
    }

    // -----------------------------------------------------------------------
    // 4. 计时：GEMV(100 次) / 带宽 / dp4a 加速比
    // -----------------------------------------------------------------------
    section("4. GEMV 计时 / 有效带宽 / dp4a 加速比");
    const double w_bytes = static_cast<double>(w.bytes());
    auto launch_hw = [&] { nova::syclk::gemv<K, true>(w, act.view(), d_out, dev.q()); };
    auto launch_sw = [&] { nova::syclk::gemv<K, false>(w, act.view(), d_out, dev.q()); };

    for (int i = 0; i < args.warmup; ++i) launch_hw();
    dev.sync();

    const double t_hw_n  = avg_launch(launch_hw, args.iters, dev.q());
    const double t_hw_2n = avg_launch(launch_hw, 2 * args.iters, dev.q());
    const double t_sw_n  = avg_launch(launch_sw, args.iters, dev.q());
    const double t_sw_2n = avg_launch(launch_sw, 2 * args.iters, dev.q());

    const double kern_hw = t_hw_2n * 2.0 - t_hw_n;   // 差分：纯 kernel 时间
    const double kern_sw = t_sw_2n * 2.0 - t_sw_n;
    const double bw_hw   = w_bytes / t_hw_n  / 1e9;
    const double bw_hw_k = (kern_hw > 1e-9) ? w_bytes / kern_hw / 1e9 : 0.0;
    const double bw_sw   = w_bytes / t_sw_n  / 1e9;

    std::printf("[bench] dp4a 路径      : %s\n", syb::dp4a_mode_name(syb::kNovaDp4aMode));
    std::printf("[bench] 权重字节      : %.3f MiB\n", w_bytes / (1024.0 * 1024.0));
    rule();
    std::printf("  HW=true  (dp4a)   %8d 次: 平均 %8.2f us/次  ->  %6.2f GB/s\n",
                args.iters, t_hw_n * 1e6, bw_hw);
    std::printf("  HW=false (标量)   %8d 次: 平均 %8.2f us/次  ->  %6.2f GB/s\n",
                args.iters, t_sw_n * 1e6, bw_sw);
    if (kern_hw > 1e-9 && kern_sw > 1e-9) {
        std::printf("  差分去 host 开销  : dp4a %6.2f us/次 (%.2f GB/s)  |  标量 %6.2f us/次 (%.2f GB/s)\n",
                    kern_hw * 1e6, bw_hw_k, kern_sw * 1e6, w_bytes / kern_sw / 1e9);
    } else {
        std::printf("  差分去 host 开销  : 负值（说明 host 提交就是瓶颈，kernel 比提交还快）\n");
    }
    rule();
    const double dp4a_ratio = (t_hw_n > 1e-12) ? (t_sw_n / t_hw_n) : 0.0;
    std::printf("[bench] dp4a 加速比 = 标量 %.2f us / dp4a %.2f us = %.2fx   门槛 >= 1.8x -> %s\n",
                t_sw_n * 1e6, t_hw_n * 1e6, dp4a_ratio,
                (dp4a_ratio >= 1.8) ? "PASS" : "FAIL");
    std::printf("[bench] 有效带宽     = %.2f GB/s（设计带宽 %.1f 的 %.0f%%）门槛 >= 70 GB/s -> %s\n",
                bw_hw, syb::profile::kDesignBandwidthGBs,
                100.0 * bw_hw / syb::profile::kDesignBandwidthGBs,
                (bw_hw >= 70.0) ? "PASS" : "FAIL");
    if (bw_hw > syb::profile::kTheoreticalBandwidthGBs * 1.05) {
        std::printf("[bench] 注意: 带宽超过理论值 %.1f GB/s —— 7.2 MiB 的权重整个驻留在 L3，"
                    "这个数不是 DRAM 带宽。看下面的冷缓存测量。\n", syb::profile::kTheoreticalBandwidthGBs);
    }

    // -----------------------------------------------------------------------
    // 4b. dp4a 指令级微基准（= docs/BACKEND_INTEL_255H.md 的 E1）
    // -----------------------------------------------------------------------
    double micro_ratio = 0.0;
    {
        const int micro_items = 65536;
        const int micro_iters = 4096;
        std::int32_t* dm = static_cast<std::int32_t*>(
            dev.alloc(static_cast<std::uint64_t>(micro_items) * 4, MemTag::Io, "bench.micro"));
        auto m_hw = [&] { nova::syclk::dp4a_micro(true,  micro_iters, micro_items, dm, dev.q()); };
        auto m_sw = [&] { nova::syclk::dp4a_micro(false, micro_iters, micro_items, dm, dev.q()); };
        for (int i = 0; i < 3; ++i) m_hw();
        dev.sync();
        const double tm_hw = avg_launch(m_hw, 10, dev.q());
        const double tm_sw = avg_launch(m_sw, 10, dev.q());
        const double n_dot = 8.0 * static_cast<double>(micro_iters) * static_cast<double>(micro_items);
        micro_ratio = (tm_hw > 1e-12) ? (tm_sw / tm_hw) : 0.0;
        std::printf("[bench] 微基准（纯寄存器，%d 项 x %d 迭代）:\n", micro_items, micro_iters);
        std::printf("        dp4a  %.3f ms -> %.1f G dp4a/s\n", tm_hw * 1e3, n_dot / tm_hw / 1e9);
        std::printf("        标量   %.3f ms -> %.1f G MAC/s\n", tm_sw * 1e3, n_dot * 4.0 / tm_sw / 1e9);
        std::printf("        指令级加速比 = %.2fx  （≈1 说明 dp4a 被展开成 4 条乘加；>=2.5 说明是单条指令）\n",
                    micro_ratio);
        dev.release(dm, MemTag::Io, static_cast<std::uint64_t>(micro_items) * 4);
    }

    // -----------------------------------------------------------------------
    // 5. 冷缓存（绕开 L3）—— 可选
    // -----------------------------------------------------------------------
    double bw_cold = 0.0;
    if (args.cold) {
        section("5. 冷缓存带宽（每轮先冲掉 L3）");
        const std::uint64_t flush_bytes = 64ull * 1024 * 1024;
        void* flush = nullptr;
        try {
            flush = dev.alloc(flush_bytes, MemTag::Scratch, "bench.flush");
        } catch (...) {
            flush = nullptr;
        }
        if (flush == nullptr) {
            std::printf("[bench] 跳过：%.0f MiB 的 flush buffer 分配失败\n",
                        static_cast<double>(flush_bytes) / (1024.0 * 1024.0));
        } else {
            auto flush_only = [&] { dev.q().memset(flush, 0, flush_bytes); };
            auto flush_gemv = [&] {
                dev.q().memset(flush, 0, flush_bytes);
                nova::syclk::gemv<K, true>(w, act.view(), d_out, dev.q());
            };
            const double t_f  = avg_launch(flush_only, args.iters, dev.q());
            const double t_fg = avg_launch(flush_gemv, args.iters, dev.q());
            const double t_g  = t_fg - t_f;
            bw_cold = (t_g > 1e-9) ? w_bytes / t_g / 1e9 : 0.0;
            std::printf("[bench] flush 单独 %.2f ms/次；flush+gemv %.2f ms/次；差值 = gemv %.2f us -> %.2f GB/s\n",
                        t_f * 1e3, t_fg * 1e3, t_g * 1e6, bw_cold);
            std::printf("[bench] 结论：%s\n", (bw_cold >= 70.0)
                            ? "冷缓存下也过 70 GB/s（说明 100 GB/s 设计带宽可达）"
                            : "冷缓存下不达标 —— 真实 7B 推理会按这个数走，见 docs/SYCL_PORT.md §7.1");
            dev.release(flush, MemTag::Scratch, flush_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // 6. gemv4（四列共享一次权重读）
    // -----------------------------------------------------------------------
    section("6. gemv4 四列批量");
    const int ncols4 = 4;
    ActBuf act4 = alloc_act(dev, ncols4, K, "bench.act4");
    {
        std::vector<float> hx4(static_cast<std::size_t>(ncols4) * K);
        Rng rng;
        for (std::size_t i = 0; i < hx4.size(); ++i) hx4[i] = rng.next() * 2.5f;
        // 第 0 列故意与单列 GEMV 用同一组激活，后面拿它做逐位对拍
        for (int i = 0; i < K; ++i) hx4[static_cast<std::size_t>(i)] = host_x[static_cast<std::size_t>(i)];
        float* dx4 = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(ncols4) * K * 4, MemTag::Io, "bench.x4"));
        dev.upload(dx4, hx4.data(), static_cast<std::uint64_t>(ncols4) * K * 4);
        nova::syclk::quant_rows<K>(act4.view(), dx4, ncols4, dev.q());
        dev.sync();
        dev.release(dx4, MemTag::Io, static_cast<std::uint64_t>(ncols4) * K * 4);
    }
    float* d_out4 = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(ncols4) * rows * 4, MemTag::Io, "bench.out4"));
    auto launch4 = [&] {
        nova::syclk::gemv4<K, true>(w, act4.qs, act4.d, act4.s, d_out4, dev.q());
    };
    for (int i = 0; i < args.warmup; ++i) launch4();
    dev.sync();
    const double t4 = avg_launch(launch4, args.iters, dev.q());
    const double bw4 = w_bytes / t4 / 1e9;
    std::printf("[bench] 1 次 gemv4 = 4 列输出；权重字节 %.3f MiB，平均 %.2f us -> 权重带宽 %.2f GB/s\n",
                w_bytes / (1024.0 * 1024.0), t4 * 1e6, bw4);
    std::printf("[bench] 折合单列 %.2f us（若比单列 GEMV 的 %.2f us 小，说明批量确实摊薄了权重读）\n",
                t4 * 1e6 / 4.0, t_hw_n * 1e6);
    // 正确性抽查：gemv4 的第 0 列用的是与单列 GEMV 完全相同的激活，两者应当逐位相同
    {
        std::vector<float> h0(static_cast<std::size_t>(rows)), h4(static_cast<std::size_t>(rows) * ncols4);
        nova::syclk::gemv<K, true>(w, act.view(), d_out, dev.q());
        dev.sync();
        dev.download(h0.data(), d_out, static_cast<std::uint64_t>(rows) * 4);
        dev.download(h4.data(), d_out4, static_cast<std::uint64_t>(ncols4) * rows * 4);
        const ErrStat e4 = compare_f32(h4.data(), h0.data(), rows);
        std::printf("[bench] gemv4 第 0 列 vs 单列 gemv: rel_l2 = %.3e  max_abs = %.3e  %s\n",
                    e4.rel_l2, e4.max_abs, (e4.rel_l2 < 1e-6) ? "ok" : "FAIL");
        check_fail += (e4.rel_l2 < 1e-6) ? 0 : 1;
    }

    // -----------------------------------------------------------------------
    // 7. 推算 decode tok/s
    // -----------------------------------------------------------------------
    section("7. 推算 decode tok/s（带宽模型）");
    struct ModelRow { const char* name; double bytes; int L; int nkv; int hd; };
    const ModelRow models[] = {
        {"Qwen2.5-1.5B", static_cast<double>(nova::Qwen25_1p5B::weight_bytes()),
         nova::Qwen25_1p5B::L, nova::Qwen25_1p5B::NKV, nova::Qwen25_1p5B::HD},
        {"Qwen2.5-7B", static_cast<double>(nova::Qwen25_7B::weight_bytes()),
         nova::Qwen25_7B::L, nova::Qwen25_7B::NKV, nova::Qwen25_7B::HD},
    };
    const double bw_use = bw_hw;   // 门槛口径：L3 热态实测带宽
    std::printf("  带宽取值: %.2f GB/s（L3 热态实测）", bw_use);
    if (bw_cold > 0.0) std::printf("；冷态 %.2f GB/s 也一并列出", bw_cold);
    std::printf("\n\n");
    std::printf("  %-14s %10s %8s %10s %10s %10s\n", "模型", "权重 MiB", "ctx", "含KV MiB", "tok/s(热)", "tok/s(冷)");
    rule();
    double tps_15b = 0.0;
    double tps_7b  = 0.0;
    for (const auto& m : models) {
        for (int ctx : {args.ctx, 4096}) {
            const double kv = 2.0 * m.L * ctx * m.nkv * m.hd * 4.0;   // 每 token 读一遍 KV
            const double tot = m.bytes + kv;
            const double tps_hot = bw_use * 1e9 / tot;
            const double tps_cold = (bw_cold > 0.0) ? bw_cold * 1e9 / tot : 0.0;
            if (std::string(m.name) == "Qwen2.5-1.5B" && ctx == args.ctx) tps_15b = tps_hot;
            if (std::string(m.name) == "Qwen2.5-7B"   && ctx == args.ctx) tps_7b  = tps_hot;
            std::printf("  %-14s %10.1f %8d %10.1f %10.1f %10.1f\n", m.name, m.bytes / (1024.0 * 1024.0), ctx,
                        kv / (1024.0 * 1024.0), tps_hot, tps_cold);
        }
    }
    rule();
    std::printf("  说明: tok/s = 带宽 / (权重字节 + 每 token KV 字节)；KV 按 fp32 记（本引擎 KV 是 fp32）。\n");
    std::printf("        没算 dispatch 开销与温降；端到端实测通常再乘 0.6~0.8（docs/SYCL_PORT.md §6.3）。\n");
    std::printf("        端到端打折后 1.5B ≈ %.0f~%.0f tok/s，7B ≈ %.1f~%.1f tok/s（ctx=%d）\n",
                tps_15b * 0.6, tps_15b * 0.8, tps_7b * 0.6, tps_7b * 0.8, args.ctx);

    // -----------------------------------------------------------------------
    // 8. 其它算子数值抽检
    // -----------------------------------------------------------------------
    if (args.checks) {
        section("8. 其它算子数值抽检（对 CPU 参考）");
        // 8.1 rms_norm
        {
            const int R = 2;
            std::vector<float> hx(static_cast<std::size_t>(R) * K), hw(static_cast<std::size_t>(K));
            Rng rng;
            for (auto& v : hx) v = rng.next() * 3.f;
            for (auto& v : hw) v = 1.f + 0.1f * rng.next();
            float* dxr = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(R) * K * 4, MemTag::Io, "chk.x"));
            float* dwr = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(K) * 4, MemTag::Io, "chk.w"));
            float* dyr = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(R) * K * 4, MemTag::Io, "chk.y"));
            dev.upload(dxr, hx.data(), static_cast<std::uint64_t>(R) * K * 4);
            dev.upload(dwr, hw.data(), static_cast<std::uint64_t>(K) * 4);
            nova::syclk::rms_norm(dyr, dxr, dwr, R, dev.q());
            dev.sync();
            std::vector<float> hy(static_cast<std::size_t>(R) * K), href(static_cast<std::size_t>(R) * K);
            dev.download(hy.data(), dyr, static_cast<std::uint64_t>(R) * K * 4);
            for (int r = 0; r < R; ++r) {
                double ss = 0.0;
                for (int i = 0; i < K; ++i) {
                    const double v = hx[static_cast<std::size_t>(r) * K + i];
                    ss += v * v;
                }
                const float sc = static_cast<float>(std::sqrt(ss / K + nova::syclk::kEps));
                for (int i = 0; i < K; ++i) {
                    href[static_cast<std::size_t>(r) * K + i] =
                        static_cast<float>(hx[static_cast<std::size_t>(r) * K + i] / sc * hw[static_cast<std::size_t>(i)]);
                }
            }
            const ErrStat e = compare_f32(hy.data(), href.data(), R * K);
            std::printf("  rms_norm      rel_l2 = %.3e  max_abs = %.3e  %s\n", e.rel_l2, e.max_abs,
                        (e.rel_l2 < 1e-5) ? "ok" : "FAIL");
            check_fail += (e.rel_l2 < 1e-5) ? 0 : 1;
            dev.release(dxr, MemTag::Io, static_cast<std::uint64_t>(R) * K * 4);
            dev.release(dwr, MemTag::Io, static_cast<std::uint64_t>(K) * 4);
            dev.release(dyr, MemTag::Io, static_cast<std::uint64_t>(R) * K * 4);
        }
        // 8.2 rope（NeoX 与交错各跑一次；只有被编译进来的那种能跑）
        {
            const int heads = nova::syclk::kNH + nova::syclk::kNKV;
            const int HD = nova::syclk::kHD;
            const int half = HD / 2;
            const std::int64_t pos = 7;
            std::vector<float> hq(static_cast<std::size_t>(heads) * HD), hk(static_cast<std::size_t>(heads) * HD);
            Rng rng;
            for (auto& v : hq) v = rng.next();
            for (auto& v : hk) v = rng.next() * 0.5f;
            float* dq = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(heads) * HD * 4, MemTag::Io, "chk.q"));
            float* dk = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(heads) * HD * 4, MemTag::Io, "chk.k"));
            float* dcs = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(nova::kCtx) * half * 4, MemTag::Io, "chk.cs"));
            float* dsn = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(nova::kCtx) * half * 4, MemTag::Io, "chk.sn"));
            dev.upload(dq, hq.data(), static_cast<std::uint64_t>(heads) * HD * 4);
            dev.upload(dk, hk.data(), static_cast<std::uint64_t>(heads) * HD * 4);
            nova::syclk::rope_table(dcs, dsn, dev.q());
            dev.sync();
            std::vector<float> hcs(static_cast<std::size_t>(nova::kCtx) * half), hsn(hcs.size());
            dev.download(hcs.data(), dcs, static_cast<std::uint64_t>(nova::kCtx) * half * 4);
            dev.download(hsn.data(), dsn, static_cast<std::uint64_t>(nova::kCtx) * half * 4);
            nova::syclk::rope_apply(dq, dk, pos, dcs, dsn, dev.q());
            dev.sync();
            std::vector<float> gq(static_cast<std::size_t>(heads) * HD), gk(gq.size());
            dev.download(gq.data(), dq, static_cast<std::uint64_t>(heads) * HD * 4);
            dev.download(gk.data(), dk, static_cast<std::uint64_t>(heads) * HD * 4);
            // CPU 参考：q 部分 heads 0..NH-1，k 部分 NH..heads-1
            std::vector<float> rq = hq, rk = hk;
            for (int head = 0; head < heads; ++head) {
                float* base = (head < nova::syclk::kNH) ? (rq.data() + static_cast<std::size_t>(head) * HD)
                                                        : (rk.data() + static_cast<std::size_t>(head - nova::syclk::kNH) * HD);
                for (int i = 0; i < half; ++i) {
                    const float c = hcs[static_cast<std::size_t>(pos) * half + i];
                    const float s = hsn[static_cast<std::size_t>(pos) * half + i];
                    const int a = nova::syclk::kRopeNeox ? i : (2 * i);
                    const int b = nova::syclk::kRopeNeox ? (i + half) : (2 * i + 1);
                    const float x0 = base[a], x1 = base[b];
                    base[a] = x0 * c - x1 * s;
                    base[b] = x0 * s + x1 * c;
                }
            }
            const int nq = nova::syclk::kNH * HD, nk = nova::syclk::kNKV * HD;
            const ErrStat eq = compare_f32(gq.data(), rq.data(), nq);
            const ErrStat ek = compare_f32(gk.data(), rk.data(), nk);
            std::printf("  rope(%s) rel_l2 = %.3e / %.3e  max_abs = %.3e / %.3e  %s\n",
                        nova::syclk::kRopeNeox ? "NeoX" : "Norm", eq.rel_l2, ek.rel_l2, eq.max_abs, ek.max_abs,
                        (eq.rel_l2 < 1e-5 && ek.rel_l2 < 1e-5) ? "ok" : "FAIL");
            check_fail += (eq.rel_l2 < 1e-5 && ek.rel_l2 < 1e-5) ? 0 : 1;
            dev.release(dq, MemTag::Io, static_cast<std::uint64_t>(heads) * HD * 4);
            dev.release(dk, MemTag::Io, static_cast<std::uint64_t>(heads) * HD * 4);
            dev.release(dcs, MemTag::Io, static_cast<std::uint64_t>(nova::kCtx) * half * 4);
            dev.release(dsn, MemTag::Io, static_cast<std::uint64_t>(nova::kCtx) * half * 4);
        }
        // 8.3 quant：逐元素精确对拍
        {
            const int R = 3;
            std::vector<float> hx(static_cast<std::size_t>(R) * K);
            Rng rng;
            for (auto& v : hx) v = rng.next() * 4.f;
            float* dxq = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(R) * K * 4, MemTag::Io, "chk.xq"));
            dev.upload(dxq, hx.data(), static_cast<std::uint64_t>(R) * K * 4);
            ActBuf a = alloc_act(dev, R, K, "chk.act");
            nova::syclk::quant_rows<K>(a.view(), dxq, R, dev.q());
            dev.sync();
            std::vector<std::int8_t> gq(static_cast<std::size_t>(R) * K);
            std::vector<float> gd(static_cast<std::size_t>(R) * nb_d);
            std::vector<std::int32_t> gs(static_cast<std::size_t>(R) * nb_s);
            dev.download(gq.data(), a.qs, static_cast<std::uint64_t>(R) * K);
            dev.download(gd.data(), a.d, static_cast<std::uint64_t>(R) * nb_d * 4);
            dev.download(gs.data(), a.s, static_cast<std::uint64_t>(R) * nb_s * 4);
            int bad_q = 0, bad_s = 0;
            double max_ds = 0.0;
            for (int r = 0; r < R; ++r) {
                for (int b = 0; b < nb_d; ++b) {
                    const float* xp = hx.data() + static_cast<std::size_t>(r) * K + b * 32;
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xp[i]));
                    const float sc = amax * (1.f / 127.f);
                    const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
                    int s0 = 0, s1 = 0;
                    for (int i = 0; i < 32; ++i) {
                        const int q = static_cast<int>(std::nearbyint(xp[i] * inv));
                        if (q != static_cast<int>(gq[static_cast<std::size_t>(r) * K + b * 32 + i])) ++bad_q;
                        (i < 16 ? s0 : s1) += q;
                    }
                    max_ds = std::fmax(max_ds, std::fabs(static_cast<double>(sc) - gd[static_cast<std::size_t>(r) * nb_d + b]));
                    if (s0 != gs[(static_cast<std::size_t>(r) * nb_d + b) * 2 + 0]) ++bad_s;
                    if (s1 != gs[(static_cast<std::size_t>(r) * nb_d + b) * 2 + 1]) ++bad_s;
                }
            }
            std::printf("  quant         int8 不一致 %d 个；16-元素和 不一致 %d 个；scale max_abs = %.3e  %s\n",
                        bad_q, bad_s, max_ds, (bad_q == 0 && bad_s == 0) ? "ok" : "FAIL");
            check_fail += (bad_q == 0 && bad_s == 0) ? 0 : 1;
            dev.release(dxq, MemTag::Io, static_cast<std::uint64_t>(R) * K * 4);
            dev.release(a.qs, MemTag::Act, syb::act_q8_bytes(R, K));
            dev.release(a.d, MemTag::Act, syb::act_d_bytes(R, K));
            dev.release(a.s, MemTag::Act, syb::act_s_bytes(R, K));
        }
        // 8.4 attention：decode 与 prefill 各一次
        {
            const int NH = nova::syclk::kNH, NKV = nova::syclk::kNKV, HD = nova::syclk::kHD;
            const int64_t maxpos = 32;
            const std::size_t kvn = static_cast<std::size_t>(maxpos) * NKV * HD;
            std::vector<float> hq(static_cast<std::size_t>(NH) * HD), hkc(kvn), hvc(kvn);
            Rng rng;
            for (auto& v : hq) v = rng.next();
            for (auto& v : hkc) v = rng.next() * 0.7f;
            for (auto& v : hvc) v = rng.next() * 0.7f;
            float* dqa = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(NH) * HD * 4, MemTag::Io, "chk.qa"));
            float* dka = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(kvn) * 4, MemTag::Io, "chk.ka"));
            float* dva = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(kvn) * 4, MemTag::Io, "chk.va"));
            float* doa = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(NH) * HD * 4, MemTag::Io, "chk.oa"));
            float* dsc = static_cast<float*>(dev.alloc(
                static_cast<std::uint64_t>(nova::syclk::attn_decode_scratch_floats()) * 4, MemTag::Scratch, "chk.scratch"));
            dev.upload(dqa, hq.data(), static_cast<std::uint64_t>(NH) * HD * 4);
            dev.upload(dka, hkc.data(), static_cast<std::uint64_t>(kvn) * 4);
            dev.upload(dva, hvc.data(), static_cast<std::uint64_t>(kvn) * 4);

            const int n_past = 9;   // n_keys = 10
            nova::syclk::attn_decode(dqa, dka, dva, doa, n_past, dsc, dev.q());
            dev.sync();
            std::vector<float> go(static_cast<std::size_t>(NH) * HD), ref(static_cast<std::size_t>(NH) * HD);
            dev.download(go.data(), doa, static_cast<std::uint64_t>(NH) * HD * 4);
            const float scale = (HD == 64) ? 0.125f : 0.08838834764831845f;
            for (int h = 0; h < NH; ++h) {
                const int kvh = h / (NH / NKV);
                const int n_keys = n_past + 1;
                std::vector<double> sc(static_cast<std::size_t>(n_keys));
                double mx = -1e30;
                for (int p = 0; p < n_keys; ++p) {
                    double acc = 0.0;
                    for (int d = 0; d < HD; ++d) {
                        acc += static_cast<double>(hq[static_cast<std::size_t>(h) * HD + d]) *
                               static_cast<double>(hkc[(static_cast<std::size_t>(p) * NKV + kvh) * HD + d]);
                    }
                    sc[static_cast<std::size_t>(p)] = acc * scale;
                    if (sc[static_cast<std::size_t>(p)] > mx) mx = sc[static_cast<std::size_t>(p)];
                }
                double sum = 0.0;
                for (int p = 0; p < n_keys; ++p) { sc[static_cast<std::size_t>(p)] = std::exp(sc[static_cast<std::size_t>(p)] - mx); sum += sc[static_cast<std::size_t>(p)]; }
                for (int d = 0; d < HD; ++d) {
                    double a = 0.0;
                    for (int p = 0; p < n_keys; ++p) {
                        a += sc[static_cast<std::size_t>(p)] * static_cast<double>(hvc[(static_cast<std::size_t>(p) * NKV + kvh) * HD + d]);
                    }
                    ref[static_cast<std::size_t>(h) * HD + d] = static_cast<float>(a / sum);
                }
            }
            const ErrStat ea = compare_f32(go.data(), ref.data(), NH * HD);
            std::printf("  attn_decode   rel_l2 = %.3e  max_abs = %.3e  (n_past=%d)  %s\n", ea.rel_l2, ea.max_abs,
                        n_past, (ea.rel_l2 < 1e-5) ? "ok" : "FAIL");
            check_fail += (ea.rel_l2 < 1e-5) ? 0 : 1;

            // prefill: n_q = 3, n_past = 5
            const int nq = 3, npast = 5;
            std::vector<float> hqp(static_cast<std::size_t>(nq) * NH * HD);
            for (auto& v : hqp) v = rng.next();
            float* dqp = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(nq) * NH * HD * 4, MemTag::Io, "chk.qp"));
            float* dop = static_cast<float*>(dev.alloc(static_cast<std::uint64_t>(nq) * NH * HD * 4, MemTag::Io, "chk.op"));
            dev.upload(dqp, hqp.data(), static_cast<std::uint64_t>(nq) * NH * HD * 4);
            nova::syclk::attn_prefill(dqp, dka, dva, dop, nq, npast, dev.q());
            dev.sync();
            std::vector<float> gop(static_cast<std::size_t>(nq) * NH * HD), refp(gop.size());
            dev.download(gop.data(), dop, static_cast<std::uint64_t>(nq) * NH * HD * 4);
            for (int t = 0; t < nq; ++t) {
                const int64_t pos = npast + t;
                for (int h = 0; h < NH; ++h) {
                    const int kvh = h / (NH / NKV);
                    const int nk = static_cast<int>(pos) + 1;
                    std::vector<double> sc(static_cast<std::size_t>(nk));
                    double mx = -1e30;
                    for (int p = 0; p < nk; ++p) {
                        double acc = 0.0;
                        for (int d = 0; d < HD; ++d) {
                            acc += static_cast<double>(hqp[((static_cast<std::size_t>(t) * NH + h)) * HD + d]) *
                                   static_cast<double>(hkc[(static_cast<std::size_t>(p) * NKV + kvh) * HD + d]);
                        }
                        sc[static_cast<std::size_t>(p)] = acc * scale;
                        if (sc[static_cast<std::size_t>(p)] > mx) mx = sc[static_cast<std::size_t>(p)];
                    }
                    double sum = 0.0;
                    for (int p = 0; p < nk; ++p) { sc[static_cast<std::size_t>(p)] = std::exp(sc[static_cast<std::size_t>(p)] - mx); sum += sc[static_cast<std::size_t>(p)]; }
                    for (int d = 0; d < HD; ++d) {
                        double a = 0.0;
                        for (int p = 0; p < nk; ++p) {
                            a += sc[static_cast<std::size_t>(p)] * static_cast<double>(hvc[(static_cast<std::size_t>(p) * NKV + kvh) * HD + d]);
                        }
                        refp[((static_cast<std::size_t>(t) * NH + h)) * HD + d] = static_cast<float>(a / sum);
                    }
                }
            }
            const ErrStat ep = compare_f32(gop.data(), refp.data(), nq * NH * HD);
            std::printf("  attn_prefill  rel_l2 = %.3e  max_abs = %.3e  (n_q=%d, n_past=%d)  %s\n", ep.rel_l2, ep.max_abs,
                        nq, npast, (ep.rel_l2 < 1e-5) ? "ok" : "FAIL");
            check_fail += (ep.rel_l2 < 1e-5) ? 0 : 1;

            dev.release(dqa, MemTag::Io, static_cast<std::uint64_t>(NH) * HD * 4);
            dev.release(dka, MemTag::Io, static_cast<std::uint64_t>(kvn) * 4);
            dev.release(dva, MemTag::Io, static_cast<std::uint64_t>(kvn) * 4);
            dev.release(doa, MemTag::Io, static_cast<std::uint64_t>(NH) * HD * 4);
            dev.release(dsc, MemTag::Scratch, static_cast<std::uint64_t>(nova::syclk::attn_decode_scratch_floats()) * 4);
            dev.release(dqp, MemTag::Io, static_cast<std::uint64_t>(nq) * NH * HD * 4);
            dev.release(dop, MemTag::Io, static_cast<std::uint64_t>(nq) * NH * HD * 4);
        }
    }

    // -----------------------------------------------------------------------
    // 9. 门槛汇总
    // -----------------------------------------------------------------------
    section("9. 门槛汇总");
    int failed = 0;
    auto gate = [&](bool ok, const char* name, double got, double need) {
        std::printf("  [%s] %-34s 实测 %8.2f   门槛 %8.2f\n", ok ? "PASS" : "FAIL", name, got, need);
        if (!ok) ++failed;
    };
    gate(bw_hw >= 70.0,             "GEMV 有效带宽 (GB/s)",     bw_hw, 70.0);
    gate(dp4a_ratio >= 1.8,         "dp4a / 标量 加速比",       dp4a_ratio, 1.8);
    gate(tps_15b >= 55.0,           "1.5B decode (tok/s)",      tps_15b, 55.0);
    if (micro_ratio > 0.0) {
        std::printf("  [INFO] dp4a 指令级微基准加速比 %.2fx（不是门槛项；用于区分"
                    "「带宽受限」与「dp4a 被软件展开」）\n", micro_ratio);
    }
    if (err.rel_l2 >= 1e-4) { std::printf("  [FAIL] GEMV rel_l2 >= 1e-4 (%.3e)\n", err.rel_l2); ++failed; }
    else std::printf("  [PASS] GEMV rel_l2 = %.3e\n", err.rel_l2);
    if (check_fail > 0) { std::printf("  [FAIL] 算子抽检有 %d 项不一致\n", check_fail); }
    else std::printf("  [PASS] 算子抽检全部一致\n");
    std::printf("\n  退出码 = 未通过项数 = %d\n", failed + check_fail);

    dev.print_memory_report();
    return failed + check_fail;
}
