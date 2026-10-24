// nova_v2/tests/sycl_mapping_check.cpp — 纯 C++ 的 GEMV 映射校验器（不需要 GPU / 不需要 oneAPI）
//
// 作用：把 src/kernel/sycl/q4k_gemv.cpp 里的 lane 映射（block/c/g/o/woff/alo/ahi/slo）、
//       块内字节偏移、get_scale_min_k4、每 16 元素的和索引、以及 dx*(dd*sc*dlo - dm*mc*sm)
//       这套代数重排，逐位复刻到宿主 C++，再与"反量化 + double 累加"的参考实现对拍。
//       这一层验证了 SYCL 机制之外的全部索引算术 —— 也就是最容易写错的部分。
//
// 构建（任何 C++17 编译器都行）：
//   cl /nologo /std:c++17 /O2 /EHsc /utf-8 /Iinclude tests/sycl_mapping_check.cpp src/loader/gguf.cpp /Fe:build/sycl_mapping_check.exe
//   build\sycl_mapping_check.exe build\q4k.gguf
// 期望：rel_l2 < 1e-5（本机实测 1.0e-07）。
#include "nova/loader/gguf.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace nova;

static inline float h2f(uint16_t h) {
    const uint32_t u = (uint32_t)h;
    const uint32_t sign = (u & 0x8000u) << 16;
    const uint32_t exp = (u >> 10) & 0x1Fu;
    const uint32_t man = u & 0x3FFu;
    uint32_t bits;
    if (exp != 0u) bits = sign | ((exp + 112u) << 23) | (man << 13);
    else if (man == 0u) bits = sign;
    else { uint32_t e = 0u, m = man; while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
           bits = sign | ((113u - e) << 23) | ((m & 0x3FFu) << 13); }
    union { uint32_t u; float f; } cv; cv.u = bits; return cv.f;
}
static inline int d4_scalar(uint32_t w, const int8_t* a, int c) {
    int r = c;
    for (int i = 0; i < 4; ++i) r += (int)((w >> (8 * i)) & 0xF) * (int)a[i];
    return r;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "build/q4k.gguf";
    GgufFile g(path);
    const TensorInfo* ti = g.find("blk.0.attn_q.weight");
    if (!ti) { printf("no tensor\n"); return 1; }
    const int K = (int)ti->ne[0], R = (int)ti->ne[1];
    printf("K=%d R=%d bytes=%llu\n", K, R, (unsigned long long)ti->bytes);

    std::vector<uint8_t> w(ti->bytes);
    {
        FILE* f = fopen(path, "rb");
        _fseeki64(f, (long long)(g.data_offset() + ti->offset), SEEK_SET);
        fread(w.data(), 1, w.size(), f);
        fclose(f);
    }
    const int BPR = K / 256, NP = (BPR + 3) / 4;
    printf("BPR=%d NP=%d\n", BPR, NP);

    // 激活：确定性随机 -> 与 quant 同规则量化
    std::vector<float> x(K);
    uint64_t s = 12345;
    for (int i = 0; i < K; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x[i] = ((float)(uint32_t)(s >> 32) / 2147483648.0f - 1.0f) * 2.5f; }
    std::vector<int8_t> xq(K);
    std::vector<float> xd(K / 32);
    std::vector<int32_t> xs(K / 16);
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
        const float sc = amax * (1.f / 127.f), inv = amax > 0 ? 127.f / amax : 0.f;
        int s0 = 0, s1 = 0;
        for (int i = 0; i < 32; ++i) {
            const int q = (int)nearbyintf(x[b * 32 + i] * inv);
            xq[b * 32 + i] = (int8_t)q; (i < 16 ? s0 : s1) += q;
        }
        xd[b] = sc; xs[b * 2 + 0] = s0; xs[b * 2 + 1] = s1;
    }

    // ---- 复刻 kernel ----
    std::vector<double> got(R);
    for (int row = 0; row < R; ++row) {
        const block_q4_K* wrow = reinterpret_cast<const block_q4_K*>(w.data()) + (long long)row * BPR;
        float acc[32];
        for (int lane = 0; lane < 32; ++lane) {
            const int Lblk = lane / 8, c = lane % 8, gg = c >> 1, o = (c & 1) << 4;
            const int woff = c * 16, alo = gg * 64 + o, ahi = alo + 32, slo = 2 * gg;
            float a = 0.f;
            for (int p = 0; p < NP; ++p) {
                const int sbi = p * 4 + Lblk;
                if (sbi >= BPR) continue;
                const block_q4_K* b = wrow + sbi;
                uint32_t wv[4];
                memcpy(wv, b->qs + woff, 16);
                const int8_t* al = xq.data() + sbi * 256 + alo;
                const int8_t* ah = xq.data() + sbi * 256 + ahi;
                const int sm_lo = xs[sbi * 16 + (alo >> 4)];
                const int sm_hi = xs[sbi * 16 + (ahi >> 4)];
                const float dx0 = xd[sbi * 8 + slo], dx1 = xd[sbi * 8 + slo + 1];
                uint8_t scl, mcl, sch, mch;
                get_scale_min_k4(slo, b->scales, scl, mcl);
                get_scale_min_k4(slo + 1, b->scales, sch, mch);
                const float dd = h2f(b->d), dm = h2f(b->dmin);
                int dlo = 0, dhi = 0;
                for (int t = 0; t < 4; ++t) dlo = d4_scalar(wv[t] & 0x0F0F0F0Fu, al + 4 * t, dlo);
                for (int t = 0; t < 4; ++t) dhi = d4_scalar((wv[t] >> 4) & 0x0F0F0F0Fu, ah + 4 * t, dhi);
                a += dx0 * ((dd * scl) * (float)dlo - (dm * mcl) * (float)sm_lo);
                a += dx1 * ((dd * sch) * (float)dhi - (dm * mch) * (float)sm_hi);
            }
            acc[lane] = a;
        }
        double sum = 0; for (int i = 0; i < 32; ++i) sum += acc[i];
        got[row] = sum;
    }

    // ---- 参考：反量化 + double 累加 ----
    std::vector<float> y(K);
    double num = 0, den = 0, mx = 0;
    for (int r = 0; r < R; ++r) {
        dequant_row(w.data(), r, K, y.data());
        double ref = 0;
        for (int i = 0; i < K; ++i) ref += (double)y[i] * (double)xd[i >> 5] * (double)xq[i];
        const double d = got[r] - ref;
        num += d * d; den += ref * ref;
        if (fabs(d) > mx) mx = fabs(d);
        if (r < 3) printf("row %d: kernel=%.8f ref=%.8f rel=%.2e\n", r, got[r], ref, fabs(d) / (fabs(ref) + 1e-30));
    }
    printf("rel_l2 = %.3e   max_abs = %.3e   (%d rows)\n", sqrt(num / den), mx, R);
    return (sqrt(num / den) < 1e-5) ? 0 : 1;
}