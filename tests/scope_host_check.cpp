// nova_v2/tests/scope_host_check.cpp — NovaScope 核心在 CPU 上的自检
//
// 为什么在 CPU 上测：本机没有 oneAPI，但 NovaScope 的**统计口径和 trace 格式**是纯逻辑，
// 必须先在没有 GPU 干扰的情况下证明它算得对，再去目标机上接真事件。
//
// 这里喂一段**合成事件流**：2 个 token × 27 层 × 14 个算子 + 每 token 4 次拷贝，
// 时长/字节都按 1.5B 的真实量级构造，然后用独立实现的参考公式逐项核对：
//   - 每个算子的次数/总耗时/有效带宽；
//   - 总墙钟、GPU 忙比、拷贝次数/字节/占比、拷贝触发率；
//   - 排队气泡（submit→start > 50 µs）的计数；
//   - Chrome trace JSON 的结构（能否解析、切片数、µs 换算、字段齐全）；
//   - CSV 的列数与行数。
//
// 构建运行：build_scope_host.bat  →  build\scope_host_check.exe
// 退出码 = 失败项数。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model/sycl/scope_core.h"

namespace {

using nova::scope::Phase;
using nova::scope::Rec;
using nova::scope::Scope;
using nova::scope::Summary;

int g_failed = 0;

void check(bool ok, const char* what, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what, detail[0] ? "  " : "", detail);
    if (!ok) ++g_failed;
}

void check_near(double got, double want, double tol, const char* what) {
    const bool ok = std::fabs(got - want) <= tol;
    char d[128];
    std::snprintf(d, sizeof(d), "got %.4f want %.4f (tol %.4f)", got, want, tol);
    check(ok, what, d);
}

// ---------------------------------------------------------------------------
// 合成一段事件流：几何级数一样地构造，但每一项都能手算，方便独立核对
// ---------------------------------------------------------------------------
struct Synth {
    // 1.5B 的每层算子：名字 + 设备耗时(us) + 每 op 权重字节
    struct Op { const char* name; double us; std::int64_t bytes; };
    std::vector<Op> ops = {
        {"rmsnorm_quant",   6.0,  36ull * 1024},
        {"gemv.q",         72.0,  9ull * 1024 * 1024},
        {"gemv.k",         20.0,  9ull * 1024 * 1024 / 8},
        {"gemv.v",         20.0,  9ull * 1024 * 1024 / 8},
        {"bias_add",        2.0,  16ull * 1024},
        {"rope_apply",      3.0,  8ull * 1024},
        {"kv_store",        2.0,  8ull * 1024},
        {"attn_decode",     9.0,  8ull * 1024},
        {"quant_rows",      2.0,  8ull * 1024},
        {"gemv.o",         25.0,  9ull * 1024 * 1024 / 4},
        {"residual_add",    2.0,  8ull * 1024},
        {"gemv.gate",      25.0,  9ull * 1024 * 1024 / 3},
        {"gemv.up",        25.0,  9ull * 1024 * 1024 / 3},
        {"swiglu_ffn",      4.0,  16ull * 1024},
        {"gemv.down",      38.0,  9ull * 1024 * 1024 / 2},
    };
    // 每 token 的拷贝：H2D token id + D2H logits（608 KB）
    struct Copy { const char* name; double us; std::int64_t bytes; };
    std::vector<Copy> copies = {
        {"H2D.token",  12.0, 4},
        {"H2D.act",     8.0, 64ull * 1024},
        {"D2H.logits", 180.0, 151936ull * 4},
        {"D2H.misc",    6.0, 4ull * 1024},
    };
    int layers = 27;
    int tokens = 2;
    double gap_us = 3.0;          // 每个 op 的 submit→start 排队间隙
};

void fill(Scope& s, const Synth& sy) {
    s.enable(true);
    s.clear();
    std::int64_t t = 1000 * 1000;   // 起点：1 ms（非 0，考验时间原点换算）
    for (int tk = 0; tk < sy.tokens; ++tk) {
        s.set_ctx(tk, -1);
        for (const auto& c : sy.copies) {
            Rec& r = s.begin(c.name, Phase::Copy, c.bytes);
            r.submit_ns = t;
            r.start_ns = t + static_cast<std::int64_t>(sy.gap_us * 1000);
            r.end_ns = r.start_ns + static_cast<std::int64_t>(c.us * 1000);
            r.resolved = true;
            t = r.end_ns;
        }
        for (int l = 0; l < sy.layers; ++l) {
            s.set_ctx(tk, l);
            for (const auto& o : sy.ops) {
                Rec& r = s.begin(o.name, Phase::Compute, o.bytes);
                r.submit_ns = t;
                r.start_ns = t + static_cast<std::int64_t>(sy.gap_us * 1000);
                r.end_ns = r.start_ns + static_cast<std::int64_t>(o.us * 1000);
                r.resolved = true;
                t = r.end_ns;
            }
        }
    }
}

// 参考公式（独立实现，不复用 Scope::summarize）
struct Ref {
    double wall_ns = 0, busy_ns = 0, copy_ns = 0;
    int copy_count = 0;
    std::int64_t copy_bytes = 0;
    int gaps = 0;
    double gap_ns = 0;
    std::int64_t weight_bytes = 0;
};

Ref ref_of(const std::vector<Rec>& rs) {
    Ref f;
    // 墙钟口径 = 设备时间轴：第一个 op 的 start 到最后一个 op 的 end
    const std::int64_t t0 = rs.empty() ? 0 : rs.front().start_ns;
    std::int64_t t1 = t0;
    for (const auto& r : rs) {
        t1 = std::max(t1, r.end_ns);
        // 与实现同口径：GPU 忙只算非拷贝
        if (r.phase != Phase::Copy) f.busy_ns += static_cast<double>(r.end_ns - r.start_ns);
        const double g = static_cast<double>(r.start_ns - r.submit_ns);
        if (g > 50e3) f.gaps++;
        f.gap_ns += g;
        if (r.phase == Phase::Copy) {
            f.copy_count++;
            f.copy_bytes += r.bytes;
            f.copy_ns += static_cast<double>(r.end_ns - r.start_ns);
        } else if (r.phase == Phase::Compute || r.phase == Phase::Other) {
            f.weight_bytes += r.bytes;
        }
    }
    f.wall_ns = static_cast<double>(t1 - t0);
    return f;
}

// 极简 JSON 扫描：数 "ph":"X" 的切片、校验 ts 是否单调、确认关键字段存在
struct JsonScan {
    int slices = 0;
    int counters = 0;
    bool has_display_unit = false;
    bool has_tool = false;
    bool monotonic = true;
    bool has_args_bytes = false;
    bool has_bw = false;
    double first_ts = -1.0, last_ts = -1.0;
};

JsonScan scan_json(const std::string& j) {
    JsonScan s;
    s.has_display_unit = j.find("\"displayTimeUnit\": \"ns\"") != std::string::npos;
    s.has_tool = j.find("NovaScope") != std::string::npos;
    s.has_args_bytes = j.find("\"bytes\":") != std::string::npos;
    s.has_bw = j.find("\"bw_gbs\":") != std::string::npos;

    double prev = -1.0;
    std::size_t p = 0;
    while ((p = j.find("\"ph\":\"", p)) != std::string::npos) {
        const char ph = j[p + 6];
        if (ph == 'X') {
            ++s.slices;
            const std::size_t ts = j.find("\"ts\":", p);
            if (ts != std::string::npos) {
                const double v = std::atof(j.c_str() + ts + 5);
                if (s.first_ts < 0.0) s.first_ts = v;
                if (prev >= 0.0 && v < prev - 1e-6) s.monotonic = false;
                prev = v;
                s.last_ts = v;
            }
        } else if (ph == 'C') {
            ++s.counters;
        }
        p += 7;
    }
    return s;
}

bool file_contains(const char* path, const char* needle, int* count) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::string all;
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
    std::fclose(f);
    int c = 0;
    for (std::size_t p = 0; (p = all.find(needle, p)) != std::string::npos; p += 1) ++c;
    if (count) *count = c;
    return true;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== NovaScope 核心自检（合成事件流，无 GPU）===\n");

    Synth sy;
    Scope& s = Scope::get();

    // ---- 0. 未启用时必须完全是空转 ----
    {
        Scope::get().clear();
        Scope::get().enable(false);
        { auto g = nova::scope::OpGuard("should_not_record", Phase::Compute, 123); (void)g; }
        check(Scope::get().recs().empty(), "未启用时不记录任何事件");
        check(std::isnan(Scope::get().summarize().hw.clock_mhz), "默认频率为 n/a（不编数）");
    }

    fill(s, sy);
    const std::size_t expect_recs =
        static_cast<std::size_t>(sy.tokens) * (sy.copies.size() + sy.layers * sy.ops.size());
    check(s.recs().size() == expect_recs, "记录条数 = token × (拷贝 + 层×算子)",
          (std::to_string(s.recs().size()) + " vs " + std::to_string(expect_recs)).c_str());

    // ---- 1. 统计口径 ----
    std::printf("\n--- 1. 统计口径（与独立参考公式比对）---\n");
    const Summary sum = s.summarize();
    const Ref ref = ref_of(s.recs());
    check_near(sum.wall_ns, ref.wall_ns, 1.0, "总墙钟 ns");
    check_near(sum.busy_ns, ref.busy_ns, 1.0, "GPU 忙 ns");
    check_near(sum.copy_ns, ref.copy_ns, 1.0, "拷贝耗时 ns");
    check(sum.copy_count == ref.copy_count, "拷贝次数");
    check(sum.copy_bytes == ref.copy_bytes, "拷贝字节");
    check(sum.gaps_over_50us == ref.gaps, "排队气泡计数");
    check_near(sum.gap_total_ns, ref.gap_ns, 1.0, "排队间隙合计");
    check(sum.weight_bytes == ref.weight_bytes, "权重流量字节");

    // 手算核对：每 token 拷贝 = 12 + 8 + 180 + 6 = 206 µs；2 token = 412 µs
    check_near(sum.copy_ns / 1e3, 412.0, 0.5, "拷贝耗时手算 412 us");
    // 每层算子 = 6+72+20+20+2+3+2+9+2+25+2+25+25+4+38 = 255 µs
    const double per_layer = 255.0;
    // GPU 忙只算计算类 op（拷贝是总线时间，单独计）—— 这才是"算力空转"该用的分母
    check_near(sum.busy_ns / 1e3, per_layer * 27 * 2, 1.0, "GPU 忙手算 = 255us×27×2（不含拷贝）");
    check_near(sum.copy_per_token(), 4.0, 1e-9, "拷贝触发率 = 4 次/token");
    check_near(sum.bytes_per_token(), (4.0 + 65536 + 151936.0 * 4 + 4096.0), 1.0, "拷贝字节/token");

    // 忙比 = 忙时/墙钟；这段合成流每个 op 有 3 µs 排队间隙，所以必然 < 1，按手算核对
    {
        const double want_ratio = (per_layer * 27 * 2) / (ref.wall_ns / 1e3);
        check_near(sum.busy_ratio(), want_ratio, 1e-6, "忙比 = 忙时/墙钟（含排队间隙）");
        check(sum.busy_ratio() > 0.5 && sum.busy_ratio() <= 1.0, "忙比落在 (0.5, 1]");
    }

    // Top 算子排序：gemv.q(72µs) 应排第一；拷贝不在 Top 里
    check(!sum.ops.empty() && sum.ops[0].name == "gemv.q", "Top1 = gemv.q");
    check(sum.ops[0].count == sy.layers * sy.tokens, "gemv.q 次数 = 层×token");
    {
        double bw = sum.ops[0].bw_gbs();
        const double want = (double)(9ull * 1024 * 1024) / (72.0 * 1e-6) / 1e9;
        check_near(bw, want, 0.5, "gemv.q 有效带宽（字节/时间）");
    }
    // ops 按总耗时降序
    bool sorted = true;
    for (std::size_t i = 1; i < sum.ops.size(); ++i) {
        if (sum.ops[i - 1].total_ns < sum.ops[i].total_ns) sorted = false;
    }
    check(sorted, "算子表按总耗时降序");

    // 频率钩子：接了才有值
    {
        nova::scope::HwSample h;
        h.clock_mhz = 1800.0;
        h.temp_c = 64.5;
        s.set_hw_sample(h);
        const Summary s2 = s.summarize();
        check(s2.hw.clock_mhz == 1800.0 && s2.hw.temp_c == 64.5, "外部硬件采样钩子生效");
        s.set_hw_sample(nova::scope::HwSample{});
    }

    // ---- 2. Chrome trace JSON ----
    std::printf("\n--- 2. Chrome trace / Perfetto JSON ---\n");
    const char* trace = "build\\nova_trace.json";
    const char* csv = "build\\nova_trace.csv";
    check(s.write_trace(trace), "写 nova_trace.json");
    check(s.write_csv(csv), "写 nova_trace.csv");

    {
        FILE* f = std::fopen(trace, "rb");
        std::string j;
        if (f) {
            char buf[65536];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) j.append(buf, n);
            std::fclose(f);
        }
        check(!j.empty(), "trace 非空", (std::to_string(j.size()) + " 字节").c_str());
        const JsonScan sc = scan_json(j);
        check(sc.has_display_unit, "有 displayTimeUnit");
        check(sc.has_tool, "有工具标识");
        check(sc.slices == static_cast<int>(expect_recs), "X 切片数 = 记录数",
              (std::to_string(sc.slices) + " vs " + std::to_string(expect_recs)).c_str());
        check(sc.counters == sy.layers * sy.tokens, "计数器条数 = 层×token（每个 token 单独一条）");
        check(sc.monotonic, "ts 单调不减（Perfetto 才能排出版）");
        check(sc.has_args_bytes && sc.has_bw, "args 带 bytes / bw_gbs");
        // µs 换算：原点 = 第一条 submit，所以第一条切片的 ts 应等于它的排队间隙（3 µs）
        check_near(sc.first_ts, sy.gap_us, 1e-3, "第一条 ts = 排队间隙（µs 原点正确）");
        // 最后一个切片不应早于第一条
        check(sc.last_ts > 0.0, "最后一条 ts > 0");
        // 括号配平
        int braces = 0, brackets = 0;
        for (char c : j) {
            if (c == '{') ++braces;
            if (c == '}') --braces;
            if (c == '[') ++brackets;
            if (c == ']') --brackets;
        }
        check(braces == 0 && brackets == 0, "JSON 括号配平");
    }

    {
        int lines = 0;
        FILE* f = std::fopen(csv, "rb");
        if (f) {
            char line[512];
            while (std::fgets(line, sizeof(line), f)) ++lines;
            std::fclose(f);
        }
        check(lines == static_cast<int>(expect_recs) + 1, "CSV 行数 = 记录数 + 表头",
              (std::to_string(lines) + " 行").c_str());
    }

    // ---- 3. 判读文案（红/黄/绿）----
    std::printf("\n--- 3. 判读（终端报告）---\n");
    s.print_report(stdout);

    Scope::get().enable(false);
    Scope::get().clear();

    std::printf("\n=== 结论：失败项 = %d ===\n", g_failed);
    if (g_failed == 0) {
        std::printf("  统计口径、trace 格式、判读逻辑在 CPU 上全部对齐。\n"
                    "  下一步（需目标机）：把真 SYCL 事件接进 scope_sycl.h，跑 10 token 端到端。\n");
    }
    return g_failed;
}
