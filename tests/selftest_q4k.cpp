// selftest_q4k.cpp — 主机端对拍（不需要 GPU）
// 1) dequant_block 的 nibble 索引与 6-bit (sc,min) 解包；
// 2) GEMV 的 warp 分工（lane -> 8 个权重 / 一个 32-子块）代数上等价于浮点参考。
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include "nova/quant/q4_k.h"

using namespace nova;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL %s\n", msg); ++g_fail; } } while (0)

// 独立参考：8 个 32-元素子块，低 nibble 用 2g，高 nibble 用 2g+1
static void ref_dequant(const block_q4_K& b, float* y) {
    const float d = half_to_float(b.d), dmin = half_to_float(b.dmin);
    for (int g = 0; g < 4; ++g) {
        uint8_t s0, m0, s1, m1;
        get_scale_min_k4(2 * g, b.scales, s0, m0);
        get_scale_min_k4(2 * g + 1, b.scales, s1, m1);
        for (int l = 0; l < 32; ++l) {
            y[g * 64 + l]      = d * s0 * (b.qs[g * 32 + l] & 0xF) - dmin * m0;
            y[g * 64 + 32 + l] = d * s1 * (b.qs[g * 32 + l] >> 4)  - dmin * m1;
        }
    }
}

int main() {
    std::mt19937 rng(12345);
    block_q4_K b{};
    b.d = 0x2C00;
    b.dmin = 0x1A00;
    for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t)(rng() & 0xFF);
    for (int i = 0; i < 128; ++i) b.qs[i] = (uint8_t)(rng() & 0xFF);

    float y1[256], y2[256];
    dequant_block(&b, y1);
    ref_dequant(b, y2);
    int bad = 0;
    for (int i = 0; i < 256; ++i) if (y1[i] != y2[i]) ++bad;
    std::printf("[dequant] bit-exact: %s (%d diffs)\n", bad ? "NO" : "YES", bad);
    CHECK(bad == 0, "dequant nibble indexing");

    int8_t x[256];
    float  xd[8];
    for (int i = 0; i < 256; ++i) x[i] = (int8_t)((rng() % 255) - 127);
    for (int i = 0; i < 8; ++i)   xd[i] = 0.003f + 0.0007f * i;

    float warp_acc = 0.f;
    for (int lane = 0; lane < 32; ++lane) {
        const int j = lane >> 3, e0 = (lane & 7) * 8, hi = e0 >> 5, xb = j * 2 + hi;
        uint8_t sc, m;
        get_scale_min_k4(xb, b.scales, sc, m);
        int dot = 0, sx = 0;
        for (int t = 0; t < 8; ++t) {
            const int idx = j * 64 + e0 + t;
            const int qv = (b.qs[j * 32 + (e0 & 31) + t] >> (4 * hi)) & 0xF;
            dot += qv * x[idx];
            sx  += x[idx];
        }
        warp_acc += xd[xb] * (half_to_float(b.d) * sc * dot - half_to_float(b.dmin) * m * sx);
    }
    float ref = 0.f;
    for (int i = 0; i < 256; ++i) ref += y2[i] * (x[i] * xd[i >> 5]);
    const float rel = std::fabs(warp_acc - ref) / std::fabs(ref);
    std::printf("[gemv] warp-reduction rel=%.3e\n", rel);
    CHECK(rel < 1e-5f, "warp lane decomposition");

    std::printf(g_fail ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASS\n", g_fail);
    return g_fail != 0;
}
