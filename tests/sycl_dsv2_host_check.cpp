// nova_v2/tests/sycl_dsv2_host_check.cpp — DeepSeek-V2-Lite 的 MLA / MoE 在 CPU 上的对拍
//
// 为什么需要它：V2-Lite 没有模型文件可下（小新离线），但**算子可以先写、先验**。
// 这个测试把 mla.cpp / moe.cpp 里真正的 kernel 体通过 tests/host_sycl 的垫片跑在 CPU 上，
// 与 double 参考实现对拍。验证的是：
//   1. mla_rope_apply  : 64 维 rope 的旋转（Neox 前后半对，表索引步长 2）
//   2. mla_kv_store    : K = [nope(128) | rope(64)]、V(128) 的 cache 布局
//   3. mla_decode      : 192 维 key / 128 维 value 的 causal softmax 注意力
//   4. moe_router/topk : sigmoid 打分 + top-6 降序 + 按和归一化
//
// 构建与运行（只需 MSVC）：
//   build_dsv2_host.bat
//   build\sycl_dsv2_host_check.exe
//
// 退出码 = 未通过项数（0 = 全过）。
#include "sycl/sycl.hpp"          // 垫片必须在所有 kernel 头之前

#include <cmath>
#include <cstdio>
#include <vector>

#include "kernels.h"
#include "nova/model/deepseek_v2_lite.h"
#include "mla.cpp"
#include "moe.cpp"

namespace {

using DSV2 = nova::DeepSeekV2Lite;

struct Rng {
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    float next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<float>(static_cast<unsigned>(s >> 32) / 2147483648.0) - 1.0f;
    }
};

struct Err {
    double rel_l2 = 0.0;
    double max_abs = 0.0;
    int    argmax = 0;
};

Err vs_ref(const float* got, const double* ref, int n) {
    Err e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(got[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        if (std::fabs(d) > e.max_abs) { e.max_abs = std::fabs(d); e.argmax = i; }
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // 崩了也要看到崩在哪一节
    std::printf("=== DeepSeek-V2-Lite 算子对拍（MLA + MoE，CPU 垫片）===\n");
    std::printf("shape: L=%d D=%d NH=%d HD=%d FFN=%d V=%d | MLA q/k=%d v=%d kv_latent=%d | MoE n=%d top=%d shared=%d ffn=%d\n",
                DSV2::L, DSV2::D, DSV2::NH, DSV2::HD, DSV2::FFN, DSV2::V,
                DSV2::kQPerHead, DSV2::kVPerHead, DSV2::kKVLatent,
                DSV2::kNExpert, DSV2::kTopK, DSV2::kNShared, DSV2::kExpertFFN);

    sycl::queue q;
    int failed = 0;

    // =======================================================================
    // 1. rope：只转 q/k 的 rope 部分（每 head 64 维）
    // =======================================================================
    std::printf("\n--- 1. mla_rope_apply（64 维 Neox rope）---\n");
    {
        const int kHalf = DSV2::kQKRopeDim / 2;            // 32
        const int pos   = 5;
        std::vector<float> qv(static_cast<std::size_t>(DSV2::NH) * DSV2::kQPerHead);
        std::vector<float> kv(static_cast<std::size_t>(DSV2::NH) * DSV2::kQKRopeDim);
        Rng rng;
        for (auto& v : qv) v = rng.next();
        for (auto& v : kv) v = rng.next();

        // rope 表：与生产一致（theta^(-i/(HD/2))，HD/2 = 64），这里直接算
        std::vector<float> cs(static_cast<std::size_t>(nova::kCtx) * kHalf * 2);
        std::vector<float> sn(cs.size());
        for (int p = 0; p < nova::kCtx; ++p) {
            for (int i = 0; i < kHalf * 2; ++i) {
                const double f = std::pow(static_cast<double>(DSV2::theta),
                                          -static_cast<double>(i) / static_cast<double>(kHalf * 2));
                cs[static_cast<std::size_t>(p) * kHalf * 2 + i] = static_cast<float>(std::cos(p * f));
                sn[static_cast<std::size_t>(p) * kHalf * 2 + i] = static_cast<float>(std::sin(p * f));
            }
        }

        std::vector<float> q_got = qv, k_got = kv;
        nova::syclk::mla_rope_apply(q_got.data(), k_got.data(), pos, cs.data(), sn.data(), q);

        // 参考：对每个 head 的 32 对做 (x0*c - x1*s, x0*s + x1*c)，表索引 2i
        auto ref_rot = [&](const std::vector<float>& src, int head, int stride, int off,
                           std::vector<double>& dst) {
            for (int i = 0; i < kHalf; ++i) {
                const double c = cs[static_cast<std::size_t>(pos) * kHalf * 2 + 2 * i];
                const double s = sn[static_cast<std::size_t>(pos) * kHalf * 2 + 2 * i];
                const double x0 = src[static_cast<std::size_t>(head) * stride + off + i];
                const double x1 = src[static_cast<std::size_t>(head) * stride + off + i + kHalf];
                dst[static_cast<std::size_t>(head) * stride + off + i]        = x0 * c - x1 * s;
                dst[static_cast<std::size_t>(head) * stride + off + i + kHalf] = x0 * s + x1 * c;
            }
        };
        std::vector<double> qr(q_got.size(), 0.0), kr(k_got.size(), 0.0);
        // 没被 rope 的部分原样照抄
        for (std::size_t i = 0; i < qv.size(); ++i) qr[i] = qv[i];
        for (std::size_t i = 0; i < kv.size(); ++i) kr[i] = kv[i];
        for (int h = 0; h < DSV2::NH; ++h) {
            ref_rot(qv, h, DSV2::kQPerHead, DSV2::kQKNopeDim, qr);
            ref_rot(kv, h, DSV2::kQKRopeDim, 0, kr);
        }
        const Err eq = vs_ref(q_got.data(), qr.data(), static_cast<int>(q_got.size()));
        const Err ek = vs_ref(k_got.data(), kr.data(), static_cast<int>(k_got.size()));
        const bool ok = (eq.max_abs < 1e-6) && (ek.max_abs < 1e-6);
        std::printf("  q max_abs %.3e | k max_abs %.3e  %s\n", eq.max_abs, ek.max_abs, ok ? "PASS" : "FAIL");
        if (!ok) ++failed;
    }

    // =======================================================================
    // 2. kv_store：K = [nope(128) | rope(64)]、V(128)
    // =======================================================================
    std::printf("\n--- 2. mla_kv_store（cache 布局）---\n");
    {
        const int ctx = 4;
        const int pos = 3;
        std::vector<float> kc(static_cast<std::size_t>(ctx) * DSV2::NH * DSV2::kQPerHead);
        std::vector<float> vc(static_cast<std::size_t>(ctx) * DSV2::NH * DSV2::kVPerHead);
        std::vector<float> kn(static_cast<std::size_t>(DSV2::NH) * DSV2::kQKNopeDim);
        std::vector<float> kr(static_cast<std::size_t>(DSV2::NH) * DSV2::kQKRopeDim);
        std::vector<float> v(static_cast<std::size_t>(DSV2::NH) * DSV2::kVPerHead);
        Rng rng;
        for (auto& x : kn) x = rng.next();
        for (auto& x : kr) x = rng.next();
        for (auto& x : v)  x = rng.next();

        nova::syclk::mla_kv_store(kc.data(), vc.data(), kn.data(), kr.data(), v.data(), pos, q);

        int bad = 0;
        for (int h = 0; h < DSV2::NH; ++h) {
            for (int d = 0; d < DSV2::kQPerHead; ++d) {
                const float want = (d < DSV2::kQKNopeDim)
                                       ? kn[static_cast<std::size_t>(h) * DSV2::kQKNopeDim + d]
                                       : kr[static_cast<std::size_t>(h) * DSV2::kQKRopeDim + (d - DSV2::kQKNopeDim)];
                const float got = kc[static_cast<std::size_t>(pos) * DSV2::NH * DSV2::kQPerHead +
                                     static_cast<std::size_t>(h) * DSV2::kQPerHead + d];
                if (got != want) ++bad;
            }
            for (int d = 0; d < DSV2::kVPerHead; ++d) {
                const float want = v[static_cast<std::size_t>(h) * DSV2::kVPerHead + d];
                const float got = vc[static_cast<std::size_t>(pos) * DSV2::NH * DSV2::kVPerHead +
                                     static_cast<std::size_t>(h) * DSV2::kVPerHead + d];
                if (got != want) ++bad;
            }
        }
        // 写 pos 之外的行必须没被碰
        int touched = 0;
        for (int p = 0; p < ctx; ++p) {
            if (p == pos) continue;
            for (std::size_t i = 0; i < static_cast<std::size_t>(DSV2::NH) * DSV2::kQPerHead; ++i) {
                if (kc[static_cast<std::size_t>(p) * DSV2::NH * DSV2::kQPerHead + i] != 0.f) ++touched;
            }
        }
        const bool ok = (bad == 0) && (touched == 0);
        std::printf("  不一致 %d 个；越界写 %d 个  %s\n", bad, touched, ok ? "PASS" : "FAIL");
        if (!ok) ++failed;
    }

    // =======================================================================
    // 3. decode：192 维 key、128 维 value、causal softmax
    // =======================================================================
    std::printf("\n--- 3. mla_decode（192/128 维 causal 注意力）---\n");
    {
        const int ctx = 8;
        for (int n_past : {0, 3, 7}) {
            const int n_keys = n_past + 1;
            std::vector<float> qv(static_cast<std::size_t>(DSV2::NH) * DSV2::kQPerHead);
            std::vector<float> kc(static_cast<std::size_t>(ctx) * DSV2::NH * DSV2::kQPerHead);
            std::vector<float> vc(static_cast<std::size_t>(ctx) * DSV2::NH * DSV2::kVPerHead);
            std::vector<float> o(static_cast<std::size_t>(DSV2::NH) * DSV2::kVPerHead);
            Rng rng;
            for (auto& x : qv) x = rng.next();
            for (auto& x : kc) x = rng.next();
            for (auto& x : vc) x = rng.next();

            nova::syclk::mla_decode(qv.data(), kc.data(), vc.data(), o.data(), n_past, q);

            // 参考（double）
            const double scale = 1.0 / std::sqrt(static_cast<double>(DSV2::kQPerHead));
            std::vector<double> ref(o.size(), 0.0);
            for (int h = 0; h < DSV2::NH; ++h) {
                std::vector<double> sc(static_cast<std::size_t>(n_keys));
                double mx = -1e300;
                for (int p = 0; p < n_keys; ++p) {
                    double acc = 0.0;
                    for (int d = 0; d < DSV2::kQPerHead; ++d) {
                        acc += static_cast<double>(qv[static_cast<std::size_t>(h) * DSV2::kQPerHead + d]) *
                               static_cast<double>(kc[(static_cast<std::size_t>(p) * DSV2::NH + h) * DSV2::kQPerHead + d]);
                    }
                    sc[static_cast<std::size_t>(p)] = acc * scale;
                    if (sc[static_cast<std::size_t>(p)] > mx) mx = sc[static_cast<std::size_t>(p)];
                }
                double sum = 0.0;
                for (int p = 0; p < n_keys; ++p) {
                    sc[static_cast<std::size_t>(p)] = std::exp(sc[static_cast<std::size_t>(p)] - mx);
                    sum += sc[static_cast<std::size_t>(p)];
                }
                for (int d = 0; d < DSV2::kVPerHead; ++d) {
                    double a = 0.0;
                    for (int p = 0; p < n_keys; ++p) {
                        a += sc[static_cast<std::size_t>(p)] *
                             static_cast<double>(vc[(static_cast<std::size_t>(p) * DSV2::NH + h) * DSV2::kVPerHead + d]);
                    }
                    ref[static_cast<std::size_t>(h) * DSV2::kVPerHead + d] = a / sum;
                }
            }
            const Err e = vs_ref(o.data(), ref.data(), static_cast<int>(o.size()));
            const bool ok = (e.rel_l2 < 1e-5);
            std::printf("  n_keys=%-2d rel_l2 %.3e  max_abs %.3e  %s\n", n_keys, e.rel_l2, e.max_abs, ok ? "PASS" : "FAIL");
            if (!ok) ++failed;
        }
    }

    // =======================================================================
    // 4. MoE：sigmoid 打分 + top-6 + 归一化
    // =======================================================================
    std::printf("\n--- 4. moe_router + moe_topk ---\n");
    {
        std::vector<float> x(DSV2::D);
        std::vector<float> wr(static_cast<std::size_t>(DSV2::kNExpert) * DSV2::D);
        Rng rng;
        for (auto& v : x) v = rng.next();
        for (auto& v : wr) v = rng.next() * 0.05f;

        std::vector<float> scores(DSV2::kNExpert);
        std::vector<int>   idx(DSV2::kTopK);
        std::vector<float> w(DSV2::kTopK);
        nova::syclk::moe_router(x.data(), wr.data(), scores.data(), q);
        nova::syclk::moe_topk(scores.data(), idx.data(), w.data(), q);

        // 参考：sigmoid + 排序 + 归一化
        std::vector<double> rs(DSV2::kNExpert);
        for (int e = 0; e < DSV2::kNExpert; ++e) {
            double acc = 0.0;
            for (int i = 0; i < DSV2::D; ++i) {
                acc += static_cast<double>(x[static_cast<std::size_t>(i)]) *
                       static_cast<double>(wr[static_cast<std::size_t>(e) * DSV2::D + i]);
            }
            rs[static_cast<std::size_t>(e)] = 1.0 / (1.0 + std::exp(-acc));
        }
        const Err es = vs_ref(scores.data(), rs.data(), DSV2::kNExpert);

        std::vector<int> order(DSV2::kNExpert);
        for (int e = 0; e < DSV2::kNExpert; ++e) order[static_cast<std::size_t>(e)] = e;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return rs[static_cast<std::size_t>(a)] > rs[static_cast<std::size_t>(b)];
        });
        double sum = 0.0;
        for (int k = 0; k < DSV2::kTopK; ++k) sum += rs[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])];

        int bad_idx = 0;
        double max_dw = 0.0, wsum = 0.0;
        for (int k = 0; k < DSV2::kTopK; ++k) {
            if (idx[static_cast<std::size_t>(k)] != order[static_cast<std::size_t>(k)]) ++bad_idx;
            const double want = rs[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])] / sum;
            max_dw = std::fmax(max_dw, std::fabs(static_cast<double>(w[static_cast<std::size_t>(k)]) - want));
            wsum += w[static_cast<std::size_t>(k)];
        }
        const bool ok = (es.max_abs < 1e-5) && (bad_idx == 0) && (max_dw < 1e-6) && (std::fabs(wsum - 1.0) < 1e-6);
        std::printf("  scores max_abs %.3e | idx 不一致 %d | 权重 max_abs %.3e | sum %.6f  %s\n",
                    es.max_abs, bad_idx, max_dw, wsum, ok ? "PASS" : "FAIL");
        if (!ok) ++failed;
    }

    std::printf("\n=== 结论 ===\n  失败项 = %d\n", failed);
    if (failed == 0) {
        std::printf("  MLA 与 MoE 路由的数值已对齐（double 参考）。\n"
                    "  还没做的：MoE 专家 FFN 的调度（按 idx 调 gemv）、MLA 的 matrix-absorbed、\n"
                    "  以及把 V2-Lite 的 27 层串成 runner —— 后者需要模型文件。\n");
    }
    return failed;
}
