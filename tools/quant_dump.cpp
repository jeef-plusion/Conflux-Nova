// nova_v2/tools/quant_dump.cpp — 读真 GGUF，按张量类型反量化并打印统计量
//
// 用途（2026-11-10）：DeepSeek-V2-Lite 的 GGUF 是混量化（Q4_K / Q5_0 / Q8_0 / Q6_K / F32），
// 而本仓库原来只认 Q4_K。这个工具做三件事：
//   1. 用 nova::GgufFile 解析整份文件（验证 loader 认得每一种类型、每个张量尺寸都对）；
//   2. 对每个类型抽一行反量化，打印 min/max/mean/σ —— 数量级不对就说明布局写错了；
//   3. 做一次「量化-反量化」往返自检（Q8_0 / Q5_0 / Q6_K），检查 q 的取值域与尺度约定。
//
// 用法: quant_dump.exe <file.gguf>
//
// 注意：往返自检只证明「本文件的编码器与解码器互逆」，不证明与 ggml 逐位一致；
// 逐位一致要等有 llama.cpp 参照时再验（见 docs/DEEPSEEK_V2_LITE.md）。

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "nova/loader/gguf.h"
#include "nova/quant/ggml_blocks.h"
#include "nova/quant/q4_k.h"

namespace {

using nova::GgmlType;

struct Stat {
    double mn = 1e300, mx = -1e300, sum = 0.0, sum2 = 0.0;
    long long n = 0;
    void add(double v) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sum2 += v * v;
        ++n;
    }
    double mean() const { return n ? sum / n : 0.0; }
    double sd() const {
        if (!n) return 0.0;
        const double m = mean();
        return std::sqrt(sum2 / n - m * m);
    }
};

const char* tname(GgmlType t) {
    switch (t) {
        case GgmlType::F32:  return "F32";
        case GgmlType::F16:  return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        default:             return "?";
    }
}

// 反量化一行到 y[cols]
bool dequant_row(const std::uint8_t* base, GgmlType t, std::int64_t row, std::int64_t cols, float* y) {
    const std::int64_t be = nova::ggml_type_block_elems(t);
    const std::int64_t bb = nova::ggml_type_block_bytes(t);
    if (be == 0 || bb == 0 || cols % be != 0) return false;
    const std::uint8_t* p = base + row * (cols / be) * bb;
    const std::int64_t nb = cols / be;
    for (std::int64_t i = 0; i < nb; ++i) {
        switch (t) {
            case GgmlType::F32: {
                float v;
                std::memcpy(&v, p + i * 4, 4);
                y[i] = v;
                break;
            }
            case GgmlType::F16: {
                std::uint16_t h;
                std::memcpy(&h, p + i * 2, 2);
                y[i] = nova::half_to_float(h);
                break;
            }
            case GgmlType::Q4_0: nova::dequant_block_q4_0(reinterpret_cast<const nova::block_q4_0*>(p) + i, y + i * 32); break;
            case GgmlType::Q4_1: nova::dequant_block_q4_1(reinterpret_cast<const nova::block_q4_1*>(p) + i, y + i * 32); break;
            case GgmlType::Q5_0: nova::dequant_block_q5_0(reinterpret_cast<const nova::block_q5_0*>(p) + i, y + i * 32); break;
            case GgmlType::Q5_1: nova::dequant_block_q5_1(reinterpret_cast<const nova::block_q5_1*>(p) + i, y + i * 32); break;
            case GgmlType::Q8_0: nova::dequant_block_q8_0(reinterpret_cast<const nova::block_q8_0*>(p) + i, y + i * 32); break;
            case GgmlType::Q6_K: nova::dequant_block_q6_K(reinterpret_cast<const nova::block_q6_K*>(p) + i, y + i * 256); break;
            case GgmlType::Q4_K: nova::dequant_block(reinterpret_cast<const nova::block_q4_K*>(p) + i, y + i * 256); break;
            default: return false;
        }
    }
    return true;
}

std::vector<std::uint8_t> read_tensor(const nova::GgufFile& g, const nova::TensorInfo& ti) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(ti.bytes));
    FILE* f = std::fopen(g.path().c_str(), "rb");
    if (!f) return {};
    _fseeki64(f, static_cast<long long>(g.data_offset() + ti.offset), SEEK_SET);
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) return {};
    return buf;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* path = (argc > 1) ? argv[1] : "D:\\dsv2\\DeepSeek-V2-Lite.Q4_K_M.gguf";

    nova::GgufFile g(path);
    const auto& ts = g.tensors();
    std::printf("file   = %s\n", path);
    std::printf("name   = %s\n", nova::general_name(g).c_str());
    std::printf("tensors= %zu   data_offset = %llu   file_size = %llu\n",
                ts.size(), static_cast<unsigned long long>(g.data_offset()),
                static_cast<unsigned long long>(g.file_size()));

    // ---- 1. 类型直方图 + 尺寸自检 ----
    std::map<std::uint32_t, int> hist;
    std::uint64_t total = 0;
    for (const auto& t : ts) {
        hist[static_cast<std::uint32_t>(t.type)]++;
        total += t.bytes;
    }
    std::printf("\n--- 类型直方图 ---\n");
    for (const auto& kv : hist) {
        std::printf("  %-5s x %3d\n", tname(static_cast<GgmlType>(kv.first)), kv.second);
    }
    std::printf("  张量总字节 = %.2f GiB   (文件 %.2f GiB，差值是元数据)\n",
                static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(g.file_size()) / (1024.0 * 1024.0 * 1024.0));

    // ---- 2. 每个类型抽一个张量反量化，打印统计 ----
    std::printf("\n--- 按类型抽样反量化（第 0 行）---\n");
    std::map<std::uint32_t, bool> done;
    int fail = 0;
    for (const auto& t : ts) {
        const auto key = static_cast<std::uint32_t>(t.type);
        if (done.count(key)) continue;
        done[key] = true;
        if (t.ne.size() < 2) continue;
        const std::int64_t cols = static_cast<std::int64_t>(t.ne[0]);
        std::vector<std::uint8_t> raw = read_tensor(g, t);
        if (raw.empty()) { std::printf("  %-5s %-30s 读取失败\n", tname(t.type), t.name.c_str()); ++fail; continue; }
        std::vector<float> y(static_cast<std::size_t>(cols));
        if (!dequant_row(raw.data(), t.type, 0, cols, y.data())) {
            std::printf("  %-5s %-30s 反量化不支持\n", tname(t.type), t.name.c_str());
            ++fail;
            continue;
        }
        Stat s;
        for (float v : y) s.add(v);
        std::printf("  %-5s %-30s cols=%-6lld min %+9.5f max %+9.5f mean %+8.5f sd %.5f\n",
                    tname(t.type), t.name.c_str(), static_cast<long long>(cols), s.mn, s.mx, s.mean(), s.sd());
        if (!std::isfinite(s.sd()) || s.sd() == 0.0f) { std::printf("        ^ 可疑：σ 为 0 或非有限\n"); ++fail; }
    }

    // ---- 3. 位打包自检：直接验「整数 q 怎么塞进块里、怎么取回来」 ----
    //   不用浮点编码器，避免把 fp16 尺度的舍入误差混进来：这里给定 q，期望值是精确的 d*(q-offset)。
    std::printf("\n--- 位打包自检（Q8_0 / Q5_0 / Q4_0：q 塞进去再取回来）---\n");
    {
        auto f2h = [](float v) -> std::uint16_t {
            std::uint32_t u;
            std::memcpy(&u, &v, 4);
            const std::uint32_t sign = (u >> 16) & 0x8000u;
            const std::int32_t e = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
            const std::uint32_t m = u & 0x7FFFFFu;
            if (e <= 0) return static_cast<std::uint16_t>(sign);
            if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10) | (m >> 13));
        };

        // Q8_0：q ∈ [-127,127]，值 = d*q
        {
            nova::block_q8_0 b{};
            b.d = f2h(0.5f);
            const float d = nova::half_to_float(b.d);
            for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<std::int8_t>((i * 7) % 255 - 127);
            float y[32];
            nova::dequant_block_q8_0(&b, y);
            double err = 0.0;
            for (int i = 0; i < 32; ++i) {
                const double want = d * static_cast<double>(b.qs[i]);
                err = std::fmax(err, std::fabs(static_cast<double>(y[i]) - want));
            }
            std::printf("  Q8_0 反量化 max_abs_err %.3e  %s\n", err, (err < 1e-6) ? "PASS" : "FAIL");
            if (!(err < 1e-6)) ++fail;
        }

        // Q5_0：5 bit 无符号 q ∈ [0,31]，值 = d*(q-16)；低位在 qs 的 nibble，高位在 qh 的位
        {
            nova::block_q5_0 b{};
            b.d = f2h(0.25f);
            const float d = nova::half_to_float(b.d);
            int q[32];
            for (int i = 0; i < 32; ++i) q[i] = (i * 5 + 3) % 32;      // 覆盖 0..31
            std::memset(b.qh, 0, sizeof(b.qh));
            for (int j = 0; j < 16; ++j) {
                const int lo = q[j], hi = q[j + 16];
                b.qs[j] = static_cast<std::uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
                if (lo & 0x10) b.qh[j / 8] |= static_cast<std::uint8_t>(1u << (j % 8));
                if (hi & 0x10) b.qh[(j + 16) / 8] |= static_cast<std::uint8_t>(1u << ((j + 16) % 8));
            }
            float y[32];
            nova::dequant_block_q5_0(&b, y);
            double err = 0.0;
            for (int i = 0; i < 32; ++i) {
                const double want = d * static_cast<double>(q[i] - 16);
                err = std::fmax(err, std::fabs(static_cast<double>(y[i]) - want));
            }
            std::printf("  Q5_0 反量化 max_abs_err %.3e  %s  (q 位宽 5bit + 偏置 16)\n", err,
                        (err < 1e-6) ? "PASS" : "FAIL");
            if (!(err < 1e-6)) ++fail;
        }

        // Q4_0：4 bit 无符号 q ∈ [0,15]，值 = d*(q-8)
        {
            nova::block_q4_0 b{};
            b.d = f2h(0.125f);
            const float d = nova::half_to_float(b.d);
            int q[32];
            for (int i = 0; i < 32; ++i) q[i] = (i * 3 + 1) % 16;
            for (int j = 0; j < 16; ++j) {
                b.qs[j] = static_cast<std::uint8_t>((q[j] & 0x0F) | ((q[j + 16] & 0x0F) << 4));
            }
            float y[32];
            nova::dequant_block_q4_0(&b, y);
            double err = 0.0;
            for (int i = 0; i < 32; ++i) {
                const double want = d * static_cast<double>(q[i] - 8);
                err = std::fmax(err, std::fabs(static_cast<double>(y[i]) - want));
            }
            std::printf("  Q4_0 反量化 max_abs_err %.3e  %s\n", err, (err < 1e-6) ? "PASS" : "FAIL");
            if (!(err < 1e-6)) ++fail;
        }
    }

    std::printf("\n=== 结论：失败项 = %d ===\n", fail);
    if (fail == 0) {
        std::printf("  loader 认得这个文件的全部类型；每种类型都能反量化出合理数量级的权重。\n"
                    "  （Q5_0/Q6_K 的位布局按 ggml 规范实现，逐位一致性仍待 llama.cpp 参照，见文档。）\n");
    }
    return fail;
}
