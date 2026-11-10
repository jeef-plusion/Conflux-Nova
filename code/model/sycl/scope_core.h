// nova_v2/code/model/sycl/scope_core.h — NovaScope 的核心（纯 C++，不依赖 SYCL）
//
// 拆成两个头文件是为了能**在本机（无 oneAPI）验证**：
//   scope_core.h  —— 记录、聚合、Chrome trace JSON 输出（本文件）
//   scope_sycl.h  —— SYCL 事件探针（enable_profiling 队列 + event::get_profiling_info）
//
// 设计口径（重要，别把它当 Nsight 用）：
//   ✅ 硬数据：kernel 的 device start/end（SYCL profiling event）、submit→start 的排队间隙、
//      每次 H2D/D2H 的字节数与次数、每个 op 的有效带宽、GPU 忙比（kernel 时间 / 墙钟时间）。
//   ⚠️ 估算：计算单元占用（occupancy）= 该 kernel 的 work-group 数 × 线程数 / 理论驻留上限。
//      它回答"这次发射能不能填满机器"，不等于硬件计数器里的 EU 活跃周期。
//   ❌ 拿不到：真实 EU 利用率、GPU 频率、温度 —— Windows + iGPU 没有可靠的公开接口
//      （要 Intel Metrics Discovery API / Level Zero Sysman）。这里一律填 NaN 并打印 "n/a"，
//      留 set_hw_sample() 钩子给外部采样器接（GPU-Z / sysman / 你自己的 dll）。
//
// 数据流：Scope::op() 记录一条 Rec（提交时刻 + 事件句柄占位）-> resolve() 用事件补齐设备时间
//        -> summarize() 出统计 -> write_trace() 出 Chrome/Perfetto 能吃的 JSON。
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace nova {
namespace scope {

enum class Phase : int { Compute = 0, Copy = 1, Host = 2, Other = 3 };

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Compute: return "compute";
        case Phase::Copy:    return "copy";
        case Phase::Host:    return "host";
        default:             return "other";
    }
}

// 64-bit 单调时钟（ns）。Chrome trace 里用 µs，这里统一 ns，写出时再换算。
inline std::int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

struct Rec {
    std::string name;
    Phase       phase = Phase::Compute;
    std::int64_t submit_ns = 0;    // host 侧提交时刻
    std::int64_t start_ns  = 0;    // device 侧开始（未解析时 = submit_ns）
    std::int64_t end_ns    = 0;    // device 侧结束（未解析时 = 解析后的墙钟）
    std::int64_t bytes     = 0;    // 这次 op 搬运/读取的字节数（权重字节 or 拷贝字节）
    int          layer     = -1;
    int          tok       = -1;
    bool         resolved  = false;

    double dur_ns() const { return static_cast<double>(end_ns - start_ns); }
    double gap_ns() const { return static_cast<double>(start_ns - submit_ns); }
    double bw_gbs() const {
        const double d = dur_ns();
        return (d > 0.0 && bytes > 0) ? (static_cast<double>(bytes) / (d * 1e-9) / 1e9) : 0.0;
    }
};

// 外部硬件采样钩子：频率/温度/EU 利用率。返回 false 表示"这一项拿不到"。
struct HwSample {
    double clock_mhz = std::nan("");
    double temp_c    = std::nan("");
    double eu_busy   = std::nan("");   // 0..1，只有接了 sysman 才可能有
};

struct OpStat {
    std::string  name;
    Phase        phase = Phase::Compute;
    int          count = 0;
    double       total_ns = 0.0;
    std::int64_t bytes = 0;
    double       max_ns = 0.0;
    // 有效带宽（按总时间算，与 PERF_BASELINE 同口径）
    double bw_gbs() const { return total_ns > 0.0 && bytes > 0 ? (double)bytes / (total_ns * 1e-9) / 1e9 : 0.0; }
    double mean_us() const { return count ? total_ns / count / 1e3 : 0.0; }
};

struct Summary {
    double       wall_ns       = 0.0;   // 第一次 submit 到最后一 op 结束
    double       busy_ns       = 0.0;   // 所有 op 的 device 时间之和（= 串行队列下的占用）
    double       copy_ns       = 0.0;
    int          copy_count    = 0;
    std::int64_t copy_bytes    = 0;
    int          gaps_over_50us = 0;     // submit→start > 50 µs 的次数（时间气泡）
    double       gap_total_ns  = 0.0;
    int          op_count      = 0;
    std::int64_t tokens       = 1;
    std::vector<OpStat> ops;             // 按 total_ns 降序
    HwSample     hw;

    double busy_ratio() const { return wall_ns > 0.0 ? busy_ns / wall_ns : 0.0; }
    double copy_ratio() const { return wall_ns > 0.0 ? copy_ns / wall_ns : 0.0; }
    double copy_per_token() const { return tokens > 0 ? (double)copy_count / tokens : 0.0; }
    double bytes_per_token() const { return tokens > 0 ? (double)copy_bytes / tokens : 0.0; }
    double gbs() const { return wall_ns > 0.0 ? (double)weight_bytes / (wall_ns * 1e-9) / 1e9 : 0.0; }
    std::int64_t weight_bytes = 0;       // Compute 类 op 的 bytes 之和（≈ 每 token 的权重流量）
};

// ---------------------------------------------------------------------------
// Scope：单例。所有方法在 enabled()==false 时都是空转（不碰时钟、不分配）。
// ---------------------------------------------------------------------------
class Scope {
public:
    static Scope& get() {
        static Scope s;
        return s;
    }

    void enable(bool on) { on_ = on; }
    bool enabled() const { return on_; }

    // 当前上下文（由 runner 设）
    void set_ctx(int tok, int layer) { tok_ = tok; layer_ = layer; }
    void set_token(int t) { tok_ = t; }
    void set_layer(int l) { layer_ = l; }
    int  token() const { return tok_; }
    int  layer() const { return layer_; }

    // 记录一次 op 的提交；事件（若可用）由 scope_sycl.h 填进 pending 队列。
    Rec& begin(const char* name, Phase ph, std::int64_t bytes) {
        Rec r;
        r.name = name;
        r.phase = ph;
        r.bytes = bytes;
        r.submit_ns = now_ns();
        r.start_ns = r.submit_ns;
        r.end_ns = r.submit_ns;
        r.tok = tok_;
        r.layer = layer_;
        recs_.push_back(std::move(r));
        return recs_.back();
    }

    void finish(Rec& r) {
        if (!r.resolved) r.end_ns = now_ns();
    }

    // 拷贝（H2D/D2H 都走这里；UMA 下的隐式拷贝也应该在这里冒头）
    void copy(const char* name, std::int64_t bytes) {
        if (!on_) return;
        Rec& r = begin(name, Phase::Copy, bytes);
        r.end_ns = r.submit_ns;      // 交给 resolve() 用事件补；没有事件就记 0 时长
        r.resolved = false;
    }

    void set_hw_sample(const HwSample& h) { hw_ = h; }

    std::vector<Rec>& recs() { return recs_; }
    const std::vector<Rec>& recs() const { return recs_; }
    void clear() { recs_.clear(); }

    // 汇总（只统计 resolved 或有正时长的记录）
    Summary summarize() const {
        Summary s;
        s.hw = hw_;
        std::map<std::string, OpStat> agg;
        std::int64_t t0 = 0, t1 = 0;
        bool first = true;
        int max_tok = 0;
        for (const auto& r : recs_) {
            const double d = r.dur_ns();
            if (r.tok > max_tok) max_tok = r.tok;
            if (first) { t0 = r.submit_ns; first = false; }
            t1 = std::max(t1, r.end_ns);
            if (d <= 0.0 && r.phase != Phase::Copy) continue;

            ++s.op_count;
            // GPU 忙 = 计算类 op 的设备时间之和；拷贝是总线时间，单独记（否则会虚报忙比）
            if (r.phase != Phase::Copy) s.busy_ns += d;
            const double g = r.gap_ns();
            if (g > 50e3) ++s.gaps_over_50us;
            if (g > 0.0) s.gap_total_ns += g;

            if (r.phase == Phase::Copy) {
                ++s.copy_count;
                s.copy_bytes += r.bytes;
                s.copy_ns += d;
            } else if (r.phase == Phase::Compute || r.phase == Phase::Other) {
                s.weight_bytes += r.bytes;
            }

            OpStat& st = agg[r.name];
            st.name = r.name;
            st.phase = r.phase;
            st.count++;
            st.total_ns += d;
            st.bytes += r.bytes;
            st.max_ns = std::max(st.max_ns, d);
        }
        s.wall_ns = (t1 > t0) ? static_cast<double>(t1 - t0) : 0.0;
        s.tokens = (max_tok + 1 > 0) ? (max_tok + 1) : 1;
        for (auto& kv : agg) s.ops.push_back(kv.second);
        std::sort(s.ops.begin(), s.ops.end(),
                  [](const OpStat& a, const OpStat& b) { return a.total_ns > b.total_ns; });
        return s;
    }

    // ---- Chrome trace / Perfetto JSON ----
    bool write_trace(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "{\n  \"displayTimeUnit\": \"ns\",\n  \"otherData\": {\n");
        std::fprintf(f, "    \"tool\": \"NovaScope\",\n    \"note\": \"device times from SYCL profiling events; "
                        "hw counters (clock/temp/eu) are n/a unless an external sampler is attached\"\n");
        std::fprintf(f, "  },\n  \"traceEvents\": [\n");

        // 用第一条记录做时间原点，转成 µs
        const std::int64_t t0 = recs_.empty() ? 0 : recs_.front().submit_ns;
        bool first = true;
        auto comma = [&] { if (!first) std::fprintf(f, ",\n"); first = false; };

        // 1) 每次 op 一个 X 切片；compute 走 tid 0，copy 走 tid 1（Perfetto 里就是两条泳道）
        for (const auto& r : recs_) {
            const double ts = static_cast<double>(r.start_ns - t0) / 1e3;
            double dur = r.dur_ns() / 1e3;
            if (dur <= 0.0) dur = 0.001;
            comma();
            std::fprintf(f,
                "    {\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                "\"pid\":1,\"tid\":%d,\"args\":{\"layer\":%d,\"tok\":%d,\"bytes\":%lld,\"bw_gbs\":%.3f,"
                "\"gap_us\":%.2f}}",
                json_escape(r.name).c_str(), phase_name(r.phase), ts, dur,
                (r.phase == Phase::Copy ? 1 : 0), r.layer, r.tok,
                static_cast<long long>(r.bytes), r.bw_gbs(), r.gap_ns() / 1e3);
        }

        // 2) 每个 (token, layer) 打一条计数器：该层累计字节（Perfetto 里是折线）
        {
            std::map<std::pair<int, int>, std::int64_t> per_layer_bytes;
            std::map<std::pair<int, int>, std::int64_t> per_layer_end;
            for (const auto& r : recs_) {
                if (r.layer < 0) continue;
                const std::pair<int, int> key{r.tok, r.layer};
                per_layer_bytes[key] += r.bytes;
                per_layer_end[key] = std::max(per_layer_end[key], r.end_ns);
            }
            for (const auto& kv : per_layer_end) {
                comma();
                std::fprintf(f,
                    "    {\"name\":\"layer_bytes\",\"cat\":\"counter\",\"ph\":\"C\",\"ts\":%.3f,"
                    "\"pid\":1,\"tid\":2,\"args\":{\"tok\":%d,\"layer\":%d,\"bytes\":%lld}}",
                    static_cast<double>(kv.second - t0) / 1e3, kv.first.first, kv.first.second,
                    static_cast<long long>(per_layer_bytes[kv.first]));
            }
        }

        std::fprintf(f, "\n  ]\n}\n");
        std::fclose(f);
        return true;
    }

    // ---- 终端报告 ----
    void print_report(FILE* out = stdout) const {
        const Summary s = summarize();
        std::fprintf(out, "\n=== NovaScope 报告 ===\n");
        std::fprintf(out, "  wall            %8.3f ms   (tokens=%lld)\n", s.wall_ns / 1e6,
                     static_cast<long long>(s.tokens));
        std::fprintf(out, "  GPU 忙          %8.3f ms   (%.1f%% 的墙钟时间)\n", s.busy_ns / 1e6,
                     100.0 * s.busy_ratio());
        std::fprintf(out, "  权重流量        %8.2f MiB  (%.2f GB/s 有效)\n",
                     static_cast<double>(s.weight_bytes) / (1024.0 * 1024.0), s.gbs());
        std::fprintf(out, "  拷贝            %8d 次 / %.2f MiB / %.3f ms (%.1f%% 墙钟)\n",
                     s.copy_count, static_cast<double>(s.copy_bytes) / (1024.0 * 1024.0),
                     s.copy_ns / 1e6, 100.0 * s.copy_ratio());
        std::fprintf(out, "  拷贝触发率      %8.2f 次/token, %.2f KiB/token\n", s.copy_per_token(),
                     s.bytes_per_token() / 1024.0);
        std::fprintf(out, "  排队气泡        %8d 次 > 50us，合计 %.3f ms\n", s.gaps_over_50us,
                     s.gap_total_ns / 1e6);
        char clock_s[48], temp_s[48];
        if (std::isnan(s.hw.clock_mhz)) std::snprintf(clock_s, sizeof(clock_s), "n/a");
        else std::snprintf(clock_s, sizeof(clock_s), "%.0f MHz", s.hw.clock_mhz);
        if (std::isnan(s.hw.temp_c)) std::snprintf(temp_s, sizeof(temp_s), "n/a");
        else std::snprintf(temp_s, sizeof(temp_s), "%.1f C", s.hw.temp_c);
        std::fprintf(out, "  频率 / 温度     %8s / %s\n", clock_s, temp_s);

        std::fprintf(out, "\n  Top 算子（按总耗时）\n");
        std::fprintf(out, "    %-22s %6s %10s %8s %10s %8s\n", "op", "次数", "总 ms", "占比", "GB/s", "单次 us");
        const int top = static_cast<int>(std::min<std::size_t>(s.ops.size(), 8));
        for (int i = 0; i < top; ++i) {
            const OpStat& o = s.ops[static_cast<std::size_t>(i)];
            std::fprintf(out, "    %-22s %6d %10.3f %7.1f%% %10.2f %8.2f\n", o.name.c_str(), o.count,
                         o.total_ns / 1e6, s.busy_ns > 0 ? 100.0 * o.total_ns / s.busy_ns : 0.0,
                         o.bw_gbs(), o.mean_us());
        }

        // 红绿灯判据（与 HTML 仪表盘同一套阈值）
        std::fprintf(out, "\n  判读\n");
        if (s.copy_ratio() > 0.15) {
            std::fprintf(out, "    [红] 拷贝占墙钟 %.1f%% > 15%%：数据搬运拥堵（看 Copy 泳道）\n", 100.0 * s.copy_ratio());
        } else if (s.copy_ratio() > 0.05) {
            std::fprintf(out, "    [黄] 拷贝占墙钟 %.1f%%：偏低但值得看一眼\n", 100.0 * s.copy_ratio());
        } else {
            std::fprintf(out, "    [绿] 拷贝占墙钟 %.1f%%：搬运不是瓶颈\n", 100.0 * s.copy_ratio());
        }
        if (s.busy_ratio() < 0.70) {
            std::fprintf(out, "    [红] GPU 忙比 %.1f%% < 70%%：算力空转（看排队气泡与下发间隙）\n", 100.0 * s.busy_ratio());
        } else if (s.busy_ratio() < 0.90) {
            std::fprintf(out, "    [黄] GPU 忙比 %.1f%%：还有 %.1f%% 的空隙\n", 100.0 * s.busy_ratio(),
                         100.0 * (1.0 - s.busy_ratio()));
        } else {
            std::fprintf(out, "    [绿] GPU 忙比 %.1f%%：发射几乎无空隙\n", 100.0 * s.busy_ratio());
        }
        if (s.ops.size() > 0) {
            const OpStat& hot = s.ops.front();
            std::fprintf(out, "    [信息] 最热的算子 %s：%.1f%% 的 GPU 时间，%.2f GB/s\n", hot.name.c_str(),
                         s.busy_ns > 0 ? 100.0 * hot.total_ns / s.busy_ns : 0.0, hot.bw_gbs());
            if (hot.bw_gbs() > 0.0 && hot.bw_gbs() < 20.0) {
                std::fprintf(out, "    [红] %s 的有效带宽 %.2f GB/s < 20：它在读权重却没喂饱带宽\n",
                             hot.name.c_str(), hot.bw_gbs());
            }
        }
    }

    // 一行一条的 CSV，方便画曲线
    bool write_csv(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "name,phase,layer,tok,bytes,start_us,dur_us,gap_us,bw_gbs\n");
        const std::int64_t t0 = recs_.empty() ? 0 : recs_.front().submit_ns;
        for (const auto& r : recs_) {
            std::fprintf(f, "%s,%s,%d,%d,%lld,%.3f,%.3f,%.2f,%.3f\n", r.name.c_str(), phase_name(r.phase),
                         r.layer, r.tok, static_cast<long long>(r.bytes),
                         static_cast<double>(r.start_ns - t0) / 1e3, r.dur_ns() / 1e3,
                         r.gap_ns() / 1e3, r.bw_gbs());
        }
        std::fclose(f);
        return true;
    }

private:
    static std::string json_escape(const std::string& s) {
        std::string o;
        o.reserve(s.size() + 8);
        for (char c : s) {
            if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
            else if (static_cast<unsigned char>(c) < 0x20) { o += '?'; }
            else o.push_back(c);
        }
        return o;
    }

    bool on_ = false;
    int  tok_ = 0;
    int  layer_ = -1;
    HwSample hw_;
    std::vector<Rec> recs_;
};

// ---------------------------------------------------------------------------
// 作用域守卫：离开作用域自动 finish()（host 侧兜底；设备事件由 scope_sycl 补）
// ---------------------------------------------------------------------------
class OpGuard {
public:
    OpGuard(const char* name, Phase ph, std::int64_t bytes) {
        auto& s = Scope::get();
        if (!s.enabled()) return;
        r_ = &s.begin(name, ph, bytes);
    }
    ~OpGuard() {
        if (r_) Scope::get().finish(*r_);
    }
    Rec* raw() const { return r_; }

private:
    Rec* r_ = nullptr;
};

#define NOVA_SCOPE_CAT2(a, b) a##b
#define NOVA_SCOPE_CAT(a, b) NOVA_SCOPE_CAT2(a, b)
#define NOVA_SCOPE_OP(name, phase, bytes) \
    ::nova::scope::OpGuard NOVA_SCOPE_CAT(nova_scope_guard_, __LINE__)(name, phase, bytes)

} // namespace scope
} // namespace nova
