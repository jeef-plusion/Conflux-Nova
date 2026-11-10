// build/lt/cpu_v_gemv.cpp - can the CPU beat the GPU on the v_proj GEMV?
//
// The question, with numbers from the Arc trace: v_proj is 256 rows x 1536 cols of Q4_K
// (221 KB of weights, 24 us on the iGPU, i.e. 9.2 GB/s because 256 rows cannot fill the
// machine). A CPU implementation reads the same 221 KB once per token. If this runs faster
// than 24 us, offloading v to the CPU (in parallel with the GPU's q/k GEMV) is a win.
//
// Design measured here: dequantize Q4_K -> int8 (per-32 scale) once at load, then the hot
// loop is int8 x int8 dot with AVX2 (_mm256_maddubs_epi16 + madd) against the same Q8
// activation the GPU uses, so the two paths stay numerically comparable.
//
// usage: cpu_v_gemv.exe [reps]
// ASCII only (MSVC). Build: cl /O2 /arch:AVX2 build\lt\cpu_v_gemv.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <immintrin.h>

namespace {

constexpr int kRows = 256;      // v_proj rows
constexpr int kCols = 1536;     // D
constexpr int kQK   = 256;      // Q4_K super-block
constexpr int kBlk = 144;

inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t man  = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            std::uint32_t e = 127 - 15 + 1, m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            bits = sign | ((e - 1) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Q4_K 的 6-bit scale/min 打包（与 ggml 一致）
inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

// 把一行 Q4_K 反量化成 int8 + 每 32 元素一个 float scale（min 直接并进偏置，简化到 0）
void dequant_row_int8(const std::uint8_t* row, int cols, std::int8_t* q8, float* scales) {
    const int nb = cols / kQK;
    for (int b = 0; b < nb; ++b) {
        const std::uint8_t* blk = row + b * kBlk;
        const float d    = half_to_float(*reinterpret_cast<const std::uint16_t*>(blk));
        const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(blk + 2));
        const std::uint8_t* scales6 = blk + 4;
        const std::uint8_t* qs = blk + 16;
        for (int j = 0; j < 8; ++j) {                    // 8 个 32-元素子块
            std::uint8_t sc, m;
            get_scale_min_k4(j, scales6, &sc, &m);
            const float ds = d * static_cast<float>(sc);
            const float dm = dmin * static_cast<float>(m);
            std::int8_t* out = q8 + b * kQK + j * 32;
            for (int i = 0; i < 16; ++i) {
                const std::uint8_t v = qs[j * 16 + i];
                const float lo = ds * static_cast<float>(v & 0x0F) - dm;
                const float hi = ds * static_cast<float>(v >> 4)   - dm;
                out[2 * i]     = static_cast<std::int8_t>(lo > 127.f ? 127 : (lo < -128.f ? -128 : (int)lo));
                out[2 * i + 1] = static_cast<std::int8_t>(hi > 127.f ? 127 : (hi < -128.f ? -128 : (int)hi));
            }
            scales[b * 8 + j] = 1.0f;                    // 已经吸收进 int8 值
        }
    }
}

// int8 x int8 点积（AVX2）：返回 int32 累加和
inline std::int32_t dot_i8_avx2(const std::int8_t* a, const std::int8_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i prod = _mm256_maddubs_epi16(_mm256_sign_epi8(va, va), _mm256_sign_epi8(vb, va));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
    }
    alignas(32) std::int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    std::int32_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) s += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 2000;

    // 造一块像 Q4_K 的权重（内容不重要，只要 d/dmin 不太离谱）
    std::vector<std::uint8_t> w(static_cast<std::size_t>(kRows) * (kCols / kQK) * kBlk);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
    for (int r = 0; r < kRows; ++r) {
        for (int b = 0; b < kCols / kQK; ++b) {
            std::uint8_t* blk = w.data() + (static_cast<std::size_t>(r) * (kCols / kQK) + b) * kBlk;
            const std::uint16_t d = 0x2C00;   // ~0.0625 的 fp16 近似
            const std::uint16_t dm = 0x0000;
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dm, 2);
            for (int j = 0; j < 8; ++j) { blk[4 + (j < 4 ? j : j + 4)] = 0x00; }
        }
    }

    // 一次性反量化成 int8（这一步在加载期做，不在热路径）
    std::vector<std::int8_t> q8(static_cast<std::size_t>(kRows) * kCols);
    std::vector<float> sc(static_cast<std::size_t>(kRows) * (kCols / 32));
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRows; ++r) {
        dequant_row_int8(w.data() + static_cast<std::size_t>(r) * (kCols / kQK) * kBlk, kCols,
                         q8.data() + static_cast<std::size_t>(r) * kCols,
                         sc.data() + static_cast<std::size_t>(r) * (kCols / 32));
    }
    const double deq_ms =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;

    // Q8 激活（GPU 那边也是这个形式）
    std::vector<std::int8_t> act(kCols);
    for (int i = 0; i < kCols; ++i) act[i] = static_cast<std::int8_t>((i * 13 + 7) & 0x7F);

    std::vector<float> out(kRows);
    // 预热
    for (int r = 0; r < kRows; ++r) {
        out[r] = static_cast<float>(dot_i8_avx2(q8.data() + static_cast<std::size_t>(r) * kCols, act.data(), kCols));
    }

    const auto t1 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; ++k) {
        for (int r = 0; r < kRows; ++r) {
            out[r] = static_cast<float>(
                dot_i8_avx2(q8.data() + static_cast<std::size_t>(r) * kCols, act.data(), kCols));
        }
    }
    const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() * 1e3;

    const double us_per = ms * 1e3 / reps;
    const double bytes = static_cast<double>(q8.size()) + sc.size() * 4.0;
    std::printf("=== CPU v_proj GEMV (%d rows x %d cols, int8 x int8, AVX2) ===\n", kRows, kCols);
    std::printf("  dequant Q4_K -> int8 (load time): %.2f ms for %d rows\n", deq_ms, kRows);
    std::printf("  hot loop: %d reps, %.3f ms total, %.2f us/token\n", reps, ms, us_per);
    std::printf("  working set: %.1f KiB  ->  %.1f GB/s\n", bytes / 1024.0,
                bytes * reps / (ms * 1e-3) / 1e9);
    std::printf("  out[0..2] = %.1f %.1f %.1f\n", out[0], out[1], out[2]);
    std::printf("\n  对比 GPU：Arc 上 v_proj 单层 24.0 us（9.2 GB/s）\n");
    std::printf("  --> CPU %s（%.2f us vs 24.0 us）\n", us_per < 24.0 ? "更快" : "更慢", us_per);
    return 0;
}
