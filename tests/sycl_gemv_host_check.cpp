// nova_v2/tests/sycl_gemv_host_check.cpp — 在 CPU 上跑**真正的** GEMV kernel，做方案 C 的对拍
//
// 为什么要这个：方案 C（docs/SYCL_PORT.md §11）改的是 lane 映射，改错时速度照样好看、
//   数值直接爆（SG=16 那次 attention 就是 9.7e-08 -> 2.1e38）。目标机上跑 sycl_bench 当然最权威，
//   但仓库所在的开发机没有 Intel GPU / oneAPI，改完 kernel 连编译都做不到。
//   这个测试用 tests/host_sycl 里的最小 SYCL 垫片，把 src/kernel/sycl/q4k_gemv.cpp 的
//   kernel 体原样跑在 CPU 线程上（线程 = work-item，条件变量 = barrier），
//   再用「反量化 + double 累加」的参考实现对拍。
//
// 与 tests/sycl_mapping_check.cpp 的分工：
//   - sycl_mapping_check：用宿主 C++ **复刻**旧映射的索引算术（不需要 SYCL，但复刻可能与 kernel 走样）；
//   - 本文件：**直接执行 kernel 源码**，覆盖 方案 C 的全部变体（SPLITS 1/2/4 × 连续/交错），
//     并且连激活的生产者（quant.cpp 的 quant_rows）一起跑。
//
// 构建与运行（只需要 MSVC，不需要 oneAPI；-DNOVA_SYCL_SG 与 kernel 编译时一致）：
//   build_sycl_host.bat 1.5b 16
//   build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf
// 手写的话就是下面这条（build_sycl_host.bat 干的就是这件事）：
//   cl /nologo /std:c++17 /O2 /EHsc /utf-8 /wd4068 /wd5030 ^
//      /Iinclude /Icode\device\sycl /Isrc\kernel\sycl /Itests\host_sycl ^
//      /DNOVA_SYCL_MODEL_QWEN25_1P5B /DNOVA_SYCL_DP4A=0 /DNOVA_SYCL_SG=16 ^
//      /DNOVA_SYCL_GEMV_ROWS=8 /DNOVA_SYCL_GEMV_SPLIT=4 ^
//      tests\sycl_gemv_host_check.cpp src\kernel\sycl\quant.cpp src\loader\gguf.cpp ^
//      /Fe:build\sycl_gemv_host_check.exe
//   build\sycl_gemv_host_check.exe build\q4k_15b.gguf
//
// 退出码 = 未通过的项数（0 = 全过）。
#include "sycl/sycl.hpp"          // 垫片，必须在所有 kernel 头之前（它定义了 __builtin_memcpy）

#include "nova/loader/gguf.h"
#include "nova/quant/q4_k.h"

#include "kernels.h"
#include "q4k_gemv.cpp"           // 直接把 kernel 源拿进来：同一 TU 才能调到匿名命名空间里的 launcher
#include "q4k_gemv_soa.cpp"     // SoA 内核（kernels.h 的 repack + gemv_soa 在这里被实例化）

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using nova::syclk::ActQ8View;
using nova::syclk::div_up;

struct Rng {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const std::uint32_t u = static_cast<std::uint32_t>(s >> 32);
        return (static_cast<float>(u) / 2147483648.0f) - 1.0f;
    }
};

// 一次 variant 的运行函数（由模板生成，kernel 的 SPLITS/STRIPE 是编译期常量）
using RunFn = void (*)(const std::uint8_t* w, const ActQ8View& a, float* out, int rows);

struct Variant {
    const char* name;
    int         splits;
    bool        stripe;
    RunFn       run;
};

} // namespace

namespace nova {
namespace syclk {

// 把 launcher 包一层：垫片里 queue 是同步的，出函数就代表算完
template <int SPLITS, bool STRIPE>
void host_gemv(const std::uint8_t* w, const ActQ8View& a, float* out, int rows) {
    sycl::queue q;
    gemv_launch_one<kD, true, SPLITS, STRIPE>(w, a, out, rows, q);
}

// SoA 内核的包装：repack 到 qs/hdr 后跑 gemv_soa（垫片里同步，出函数就算完）
void host_gemv_soa(const std::uint8_t* w_aos, int rows, int K, const ActQ8View& a, float* out) {
    const int bpr = K / nova::kQK;
    std::vector<std::uint8_t> qs(static_cast<std::size_t>(rows) * static_cast<std::size_t>(nova::syclk::soa_qs_row_bytes(bpr)));
    std::vector<std::uint8_t> hdr(static_cast<std::size_t>(rows) * static_cast<std::size_t>(nova::syclk::soa_hdr_row_bytes(bpr)));
    nova::syclk::repack_q4k_soa(w_aos, rows, bpr, qs.data(), hdr.data());
    nova::syclk::SoAWeightView v{};
    v.qs = qs.data(); v.hdr = hdr.data(); v.rows = rows; v.bpr = bpr;
    sycl::queue q;
    if (K == nova::syclk::kD)        nova::syclk::gemv_soa<nova::syclk::kD, true>(v, a, out, q);
    else if (K == nova::syclk::kFFN) nova::syclk::gemv_soa<nova::syclk::kFFN, true>(v, a, out, q);
}

// 激活的生产者：用的就是生产代码（quant.cpp），不是复刻
void host_quant(ActQ8View a, const float* x, int rows) {
    sycl::queue q;
    quant_rows<kD>(a, x, rows, q);
}

// 融合版的激活生产者（norm.cpp），每层 decode 都走它
void host_rmsnorm_quant(float* y, ActQ8View a, const float* x, const float* w, int rows) {
    sycl::queue q;
    rmsnorm_quant(y, a, x, w, rows, q);
}

// FFN 中段的融合激活生产者（elem.cpp）：silu(gate)*up + 量化
void host_swiglu_quant(float* y, ActQ8View a, const float* gate, const float* up, int rows) {
    sycl::queue q;
    swiglu_quant<kD>(a, y, gate, up, rows, q);
}

} // namespace syclk
} // namespace nova

namespace {

const Variant variants_all[4] = {
    {"SPLITS=1 (旧映射)",          1, false, &nova::syclk::host_gemv<1, false>},
    {"SPLITS=2 连续段",            2, false, &nova::syclk::host_gemv<2, false>},
    {"SPLITS=4 连续段 (= 方案 C)", 4, false, &nova::syclk::host_gemv<4, false>},
    {"SPLITS=4 交错段 (实验臂)",   4, true,  &nova::syclk::host_gemv<4, true >},
};

// 参考：反量化 + double 累加（独立于 kernel 的任何代数重排）
void cpu_ref(const std::uint8_t* w, int rows, int K, const std::int8_t* qs, const float* d, double* out) {
    std::vector<float> row(static_cast<std::size_t>(K));
    for (int r = 0; r < rows; ++r) {
        nova::dequant_row(w, r, K, row.data());
        double acc = 0.0;
        for (int i = 0; i < K; ++i) {
            acc += static_cast<double>(row[static_cast<std::size_t>(i)]) *
                   static_cast<double>(d[i >> 5]) *
                   static_cast<double>(qs[i]);
        }
        out[r] = acc;
    }
}

struct Err {
    double rel_l2  = 0.0;
    double max_abs = 0.0;
    int    argmax  = 0;
};

Err vs_ref(const float* got, const double* ref, int n) {
    Err e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(got[i]) - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
        const double ad = std::fabs(diff);
        if (ad > e.max_abs) { e.max_abs = ad; e.argmax = i; }
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

Err vs_f32(const float* got, const float* ref, int n) {
    Err e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += diff * diff;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        const double ad = std::fabs(diff);
        if (ad > e.max_abs) { e.max_abs = ad; e.argmax = i; }
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

} // namespace

int main(int argc, char** argv) {
    const char* path  = (argc > 1) ? argv[1] : "build/q4k.gguf";
    const char* tname = (argc > 2) ? argv[2] : "blk.0.attn_q.weight";

    std::printf("=== 方案 C 对拍：CPU 上跑真 kernel（最小 SYCL 垫片）===\n");
    std::printf("model=%s  kD=%d  SG=%d  GEMV_ROWS=%d  GEMV_SPLIT=%d  GEMV_STRIPE=%d\n",
                kSyclModelName, nova::syclk::kD, nova::syclb::kSyclSg,
                nova::syclk::kGemvRows, nova::syclk::kGemvSplits, (int)nova::syclk::kGemvStripe);

    nova::GgufFile g(path);
    const nova::TensorInfo* ti = g.find(tname);
    if (ti == nullptr) { std::printf("FATAL: 找不到张量 %s\n", tname); return 2; }
    if (ti->type != nova::GgmlType::Q4_K) { std::printf("FATAL: 张量不是 Q4_K\n"); return 2; }
    if (ti->ne.size() != 2) { std::printf("FATAL: 张量不是二维\n"); return 2; }

    const int K    = static_cast<int>(ti->ne[0]);
    const int R    = static_cast<int>(ti->ne[1]);
    if (K != nova::syclk::kD) {
        std::printf("FATAL: 张量 K=%d 与本二进制的 kD=%d 不一致，换 -DNOVA_SYCL_MODEL_* 重编\n",
                    K, nova::syclk::kD);
        return 2;
    }
    std::printf("file=%s  tensor=%s  K=%d  R=%d  bytes=%llu\n", path, tname, K, R,
                static_cast<unsigned long long>(ti->bytes));

    std::vector<std::uint8_t> w(static_cast<std::size_t>(ti->bytes));
    {
        FILE* f = std::fopen(path, "rb");
        if (f == nullptr) { std::printf("FATAL: 打不开 %s\n", path); return 2; }
        _fseeki64(f, static_cast<long long>(g.data_offset() + ti->offset), SEEK_SET);
        if (std::fread(w.data(), 1, w.size(), f) != w.size()) { std::printf("FATAL: 读张量失败\n"); return 2; }
        std::fclose(f);
    }

    // ---- 激活：走生产代码 quant_rows（fp32 -> int8 + 每 16 元素和）----
    std::vector<float> x(static_cast<std::size_t>(K));
    {
        Rng rng;
        for (int i = 0; i < K; ++i) x[static_cast<std::size_t>(i)] = rng.next() * 2.5f;
    }
    std::vector<std::int8_t>  qs(static_cast<std::size_t>(K));
    std::vector<float>        xd(static_cast<std::size_t>(K / 32));
    std::vector<std::int32_t> xs(static_cast<std::size_t>(K / 16));
    const ActQ8View act{qs.data(), xd.data(), xs.data()};
    nova::syclk::host_quant(act, x.data(), 1);

    int pre_fail = 0;

    // ---- 0. quant 逐元素对拍（SG=16 曾经在这里挂过：每个 32 元素块只量化了前 16 个）----
    std::printf("\n--- 0. quant 逐元素对拍（SG=%d，每 lane %d 个元素）---\n",
                nova::syclb::kSyclSg, 32 / nova::syclb::kSyclSg);
    {
        const int QR = 3;
        const int NB = K / 32;
        std::vector<float> qx(static_cast<std::size_t>(QR) * K);
        Rng rng;
        for (auto& v : qx) v = rng.next() * 4.f;
        std::vector<std::int8_t>  q_qs(static_cast<std::size_t>(QR) * K);
        std::vector<float>        q_d(static_cast<std::size_t>(QR) * NB);
        std::vector<std::int32_t> q_s(static_cast<std::size_t>(QR) * NB * 2);
        const ActQ8View qa{q_qs.data(), q_d.data(), q_s.data()};
        nova::syclk::host_quant(qa, qx.data(), QR);

        int bad_q = 0, bad_s = 0;
        double max_ds = 0.0;
        for (int r = 0; r < QR; ++r) {
            for (int b = 0; b < NB; ++b) {
                const float* xp = qx.data() + static_cast<std::size_t>(r) * K + b * 32;
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xp[i]));
                const float sc  = amax * (1.f / 127.f);
                const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
                int s0 = 0, s1 = 0;
                for (int i = 0; i < 32; ++i) {
                    const int q = static_cast<int>(std::rint(xp[i] * inv));
                    if (q != static_cast<int>(q_qs[static_cast<std::size_t>(r) * K + b * 32 + i])) ++bad_q;
                    (i < 16 ? s0 : s1) += q;
                }
                const std::size_t sb = static_cast<std::size_t>(r) * NB + b;
                max_ds = std::fmax(max_ds, std::fabs(static_cast<double>(sc) - q_d[sb]));
                if (s0 != q_s[sb * 2 + 0]) ++bad_s;
                if (s1 != q_s[sb * 2 + 1]) ++bad_s;
            }
        }
        const bool ok = (bad_q == 0 && bad_s == 0);
        std::printf("  quant_rows    int8 不一致 %d；16-和 不一致 %d；scale max_abs %.3e  %s\n",
                    bad_q, bad_s, max_ds, ok ? "PASS" : "FAIL");
        if (!ok) ++pre_fail;
    }

    // ---- 0b. rmsnorm_quant：用 kernel 自己的 y 反推 q8，验证 32 个元素全被量化 ----
    std::printf("\n--- 0b. rmsnorm_quant 的 q8 自洽性 ---\n");
    {
        const int NB = K / 32;
        const float eps = nova::syclk::kEps;
        std::vector<float> rx(static_cast<std::size_t>(K)), rw(static_cast<std::size_t>(K));
        Rng rng;
        for (auto& v : rx) v = rng.next() * 3.f;
        for (auto& v : rw) v = 1.f + 0.1f * rng.next();

        std::vector<float> ry(static_cast<std::size_t>(K));
        std::vector<std::int8_t>  rqs(static_cast<std::size_t>(K));
        std::vector<float>        rd(static_cast<std::size_t>(NB));
        std::vector<std::int32_t> rs(static_cast<std::size_t>(NB) * 2);
        const ActQ8View rv{rqs.data(), rd.data(), rs.data()};
        nova::syclk::host_rmsnorm_quant(ry.data(), rv, rx.data(), rw.data(), 1);

        // (a) y 对不对：与 double 参考比
        double ss = 0.0;
        for (int i = 0; i < K; ++i) ss += static_cast<double>(rx[static_cast<std::size_t>(i)]) * rx[static_cast<std::size_t>(i)];
        const double sc_ref = 1.0 / std::sqrt(ss / static_cast<double>(K) + static_cast<double>(eps));
        std::vector<double> yr(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i) {
            yr[static_cast<std::size_t>(i)] = static_cast<double>(rx[static_cast<std::size_t>(i)]) * sc_ref *
                                              static_cast<double>(rw[static_cast<std::size_t>(i)]);
        }
        const Err ey = vs_ref(ry.data(), yr.data(), K);

        // (b) q8 自洽：d 必须等于 max|y|/127，两个 16-和必须等于对应半块 rint(y*inv) 的和
        int bad_s = 0, bad_d = 0;
        for (int b = 0; b < NB; ++b) {
            const float* yp = ry.data() + b * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(yp[i]));
            const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
            int s0 = 0, s1 = 0;
            for (int i = 0; i < 32; ++i) {
                const int q = static_cast<int>(std::rint(yp[i] * inv));
                if (q != static_cast<int>(rqs[static_cast<std::size_t>(b) * 32 + i])) ++bad_s;
                (i < 16 ? s0 : s1) += q;
            }
            if (std::fabs(static_cast<double>(amax * (1.f / 127.f)) - rd[static_cast<std::size_t>(b)]) > 1e-9) ++bad_d;
            if (s0 != rs[static_cast<std::size_t>(b) * 2 + 0]) ++bad_s;
            if (s1 != rs[static_cast<std::size_t>(b) * 2 + 1]) ++bad_s;
        }
        const bool ok = (ey.rel_l2 < 1e-6) && (bad_s == 0) && (bad_d == 0);
        std::printf("  rmsnorm_quant y rel_l2 %.3e；q8 不一致 %d；scale 不一致 %d  %s\n",
                    ey.rel_l2, bad_s, bad_d, ok ? "PASS" : "FAIL");
        if (!ok) ++pre_fail;
    }

    // ---- 0c. swiglu_quant：silu(gate)*up + 量化（FFN 中段，每层也走它）----
    std::printf("\n--- 0c. swiglu_quant 的 q8 自洽性 ---\n");
    {
        const int NB = K / 32;
        std::vector<float> hg(static_cast<std::size_t>(K)), hu(static_cast<std::size_t>(K));
        Rng rng;
        for (auto& v : hg) v = rng.next() * 3.f;
        for (auto& v : hu) v = rng.next() * 2.f;

        std::vector<float> sy(static_cast<std::size_t>(K));
        std::vector<std::int8_t>  sqs(static_cast<std::size_t>(K));
        std::vector<float>        sd(static_cast<std::size_t>(NB));
        std::vector<std::int32_t> ss_(static_cast<std::size_t>(NB) * 2);
        const ActQ8View sv{sqs.data(), sd.data(), ss_.data()};
        nova::syclk::host_swiglu_quant(sy.data(), sv, hg.data(), hu.data(), 1);

        std::vector<double> yr(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i) {
            const double gv = hg[static_cast<std::size_t>(i)];
            const double uv = hu[static_cast<std::size_t>(i)];
            yr[static_cast<std::size_t>(i)] = (gv / (1.0 + std::exp(-gv))) * uv;
        }
        const Err ey = vs_ref(sy.data(), yr.data(), K);

        int bad_s = 0, bad_d = 0;
        for (int b = 0; b < NB; ++b) {
            const float* yp = sy.data() + b * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(yp[i]));
            const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
            int s0 = 0, s1 = 0;
            for (int i = 0; i < 32; ++i) {
                const int q = static_cast<int>(std::rint(yp[i] * inv));
                if (q != static_cast<int>(sqs[static_cast<std::size_t>(b) * 32 + i])) ++bad_s;
                (i < 16 ? s0 : s1) += q;
            }
            if (std::fabs(static_cast<double>(amax * (1.f / 127.f)) - sd[static_cast<std::size_t>(b)]) > 1e-9) ++bad_d;
            if (s0 != ss_[static_cast<std::size_t>(b) * 2 + 0]) ++bad_s;
            if (s1 != ss_[static_cast<std::size_t>(b) * 2 + 1]) ++bad_s;
        }
        const bool ok = (ey.rel_l2 < 1e-5) && (bad_s == 0) && (bad_d == 0);
        std::printf("  swiglu_quant y rel_l2 %.3e；q8 不一致 %d；scale 不一致 %d  %s\n",
                    ey.rel_l2, bad_s, bad_d, ok ? "PASS" : "FAIL");
        if (!ok) ++pre_fail;
    }

    // ---- 参考值 ----
    std::vector<double> ref(static_cast<std::size_t>(R));
    cpu_ref(w.data(), R, K, qs.data(), xd.data(), ref.data());

    // ---- 变体清单 ----
    constexpr int nv = static_cast<int>(sizeof(variants_all) / sizeof(variants_all[0]));
    const Variant* variants = variants_all;

    // ---- 1. 每个变体 vs 参考 ----
    std::printf("\n--- 1. 数值对拍（门槛 rel_l2 < 1e-4）---\n");
    std::printf("  %-26s %5s %5s %4s %6s %8s %10s %10s %10s  %s\n",
                "变体", "SPLIT", "SEG", "NP", "rows/WG", "groups", "rel_l2", "max_abs", "vs基线", "判定");
    std::printf("  -----------------------------------------------------------------------------------------------------------\n");

    int failed = pre_fail;
    std::vector<std::vector<float>> outs(static_cast<std::size_t>(nv), std::vector<float>(static_cast<std::size_t>(R)));
    for (int v = 0; v < nv; ++v) {
        const Variant& va = variants[v];
        const int BPR = K / nova::syclb::kQK;
        const int SEG = div_up(BPR, va.splits);
        const int NP  = div_up(SEG, nova::syclk::kBlocksInFlight);
        const int RWS = nova::syclk::kGemvRows / va.splits;
        const int groups = div_up(R, RWS);

        va.run(w.data(), act, outs[static_cast<std::size_t>(v)].data(), R);
        const Err e = vs_ref(outs[static_cast<std::size_t>(v)].data(), ref.data(), R);
        const bool ok = (e.rel_l2 < 1e-4);
        if (!ok) ++failed;

        // 与基线（SPLITS=1）的差：只反映"求和次序不同"，不是错误
        double dmax = 0.0;
        if (v > 0) {
            const Err d = vs_f32(outs[static_cast<std::size_t>(v)].data(),
                                 outs[static_cast<std::size_t>(0)].data(), R);
            dmax = d.max_abs;
        }
        std::printf("  %-26s %5d %5d %4d %6d %8d %10.3e %10.3e %10.3e  %s%s\n",
                    va.name, va.splits, SEG, NP, RWS, groups, e.rel_l2, e.max_abs, dmax,
                    ok ? "PASS" : "FAIL",
                    v == 0 ? "  (基线)" : "");
        if (!ok) {
            std::printf("        第一个不符的行 = %d  got=%.9g  ref=%.9g\n", e.argmax,
                        static_cast<double>(outs[static_cast<std::size_t>(v)][static_cast<std::size_t>(e.argmax)]),
                        ref[static_cast<std::size_t>(e.argmax)]);
        }
    }

    // ---- 1b. SoA 内核（qs/hdr 分离布局）vs 同一个 double 参考 ----
    std::printf("\n--- 1b. SoA 布局（qs 128 B + hdr 16 B，块头按 lane 连续排）---\n");
    {
        std::vector<float> out_soa(static_cast<std::size_t>(R));
        nova::syclk::host_gemv_soa(w.data(), R, K, act, out_soa.data());
        const Err e = vs_ref(out_soa.data(), ref.data(), R);
        const bool ok = (e.rel_l2 < 1e-4);
        if (!ok) ++failed;
        const Err d = vs_f32(out_soa.data(), outs[0].data(), R);
        std::printf("  %-26s rel_l2=%10.3e  max_abs=%10.3e  vsAoS=%10.3e  %s\n",
                    "SoA (qs+hdr 分离)", e.rel_l2, e.max_abs, d.max_abs, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::printf("        第一个不符的行 = %d  got=%.9g  ref=%.9g\n", e.argmax,
                        static_cast<double>(out_soa[static_cast<std::size_t>(e.argmax)]),
                        ref[static_cast<std::size_t>(e.argmax)]);
        }
    }

    // ---- 2. 尾部守卫：rows 不是 rows-per-WG 的整数倍时，最后一个 group 里越界的行不能写 ----
    std::printf("\n--- 2. 尾部守卫（rows = R-1，最后一行不该被写；门槛 rel_l2 < 1e-4）---\n");
    for (int v = 0; v < nv; ++v) {
        const Variant& va = variants[v];
        const int RWS = nova::syclk::kGemvRows / va.splits;
        std::vector<float> out(static_cast<std::size_t>(R), -12345.0f);
        va.run(w.data(), act, out.data(), R - 1);
        // 用 2-范数比（逐行相对误差在"和为 0"的行上会误报）
        const Err e = vs_ref(out.data(), ref.data(), R - 1);
        const bool untouched = (out[static_cast<std::size_t>(R - 1)] == -12345.0f);
        const bool ok = (e.rel_l2 < 1e-4) && untouched;
        if (!ok) ++failed;
        std::printf("  %-26s rows=%d rows/WG=%d groups=%d  rel_l2=%.3e  max_abs=%.3e  末行未被写=%s  %s\n",
                    va.name, R - 1, RWS, div_up(R - 1, RWS), e.rel_l2, e.max_abs,
                    untouched ? "是" : "否", ok ? "PASS" : "FAIL");
    }

    // ---- 3. 结论 ----
    std::printf("\n=== 结论 ===\n");
    std::printf("  失败项 = %d\n", failed);
    std::printf("  注：这是 CPU 上的数值对拍，只证明 lane 映射与归约顺序对；\n");
    std::printf("      带宽收益必须在目标机（Arc 130T + oneAPI）上用 sycl_bench 量，这里量不了。\n");
    return failed;
}
