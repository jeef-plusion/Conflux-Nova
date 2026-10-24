// gpu_gemv.cu — 在真卡上跑 Kernel::q_proj<1>/<4>，与 CPU 参考逐元素对拍
#include "nova/kernel.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

using namespace nova;

static uint16_t to_half(float v) {
    __half h = __float2half(v);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

int main() {
    constexpr int K = Model::D;
    constexpr int R = Model::NH * Model::HD;
    const int nb = K / kQK;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);

    std::vector<block_q4_K> w((size_t)R * nb);
    for (auto& b : w) {
        b.d    = to_half(0.002f + (rng() % 100) * 1e-5f);
        b.dmin = to_half(0.001f);
        for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t)(rng() & 0xFF);
        for (int i = 0; i < 128; ++i) b.qs[i] = (uint8_t)(rng() & 0xFF);
    }

    const int N = 4;
    std::vector<float>  x((size_t)N * K);
    std::vector<int8_t> xq((size_t)N * K);
    std::vector<float>  xd((size_t)N * (K / 32));
    for (auto& v : x) v = uf(rng);
    for (int t = 0; t < N; ++t) {
        for (int b = 0; b < K / 32; ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[(size_t)t * K + b * 32 + i]));
            const float d = amax / 127.f;
            const float inv = d > 0 ? 1.f / d : 0.f;
            for (int i = 0; i < 32; ++i)
                xq[(size_t)t * K + b * 32 + i] = (int8_t)std::lrint(x[(size_t)t * K + b * 32 + i] * inv);
            xd[(size_t)t * (K / 32) + b] = d;
        }
    }

    // CPU 参考：fp32 反量化后点积
    std::vector<float> ref((size_t)N * R, 0.f);
    std::vector<float> row(K);
    for (int r = 0; r < R; ++r) {
        dequant_row(reinterpret_cast<const uint8_t*>(w.data()), r, K, row.data());
        for (int t = 0; t < N; ++t) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += row[k] * ((float)xq[(size_t)t * K + k] * xd[(size_t)t * (K / 32) + k / 32]);
            ref[(size_t)t * R + r] = acc;
        }
    }

    block_q4_K*   d_w = nullptr;
    int8_t*       d_q = nullptr;
    float*        d_d = nullptr;
    float*        d_y = nullptr;
    cudaMalloc(&d_w, w.size() * sizeof(block_q4_K));
    cudaMalloc(&d_q, xq.size());
    cudaMalloc(&d_d, xd.size() * sizeof(float));
    cudaMalloc(&d_y, (size_t)N * R * sizeof(float));
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, xq.data(), xq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, xd.data(), xd.size() * sizeof(float), cudaMemcpyHostToDevice);

    const Q4KWeight W{reinterpret_cast<const uint8_t*>(d_w), R, K};
    int rc = 0;

    {   // 单列
        const ActQ8 A{d_q, d_d};
        Kernel::q_proj<1>(W, A, d_y, 0);
        std::vector<float> y(R);
        cudaMemcpy(y.data(), d_y, R * sizeof(float), cudaMemcpyDeviceToHost);
        double max_abs = 0, num = 0, den = 0;
        for (int r = 0; r < R; ++r) {
            const double a = y[r], b = ref[r];
            max_abs = std::fmax(max_abs, std::fabs(a - b));
            num += (a - b) * (a - b);
            den += b * b;
        }
        const double l2 = std::sqrt(num / den);
        std::printf("[gpu] q_proj<1>  max_abs=%.3e  rel_l2=%.3e\n", max_abs, l2);
        if (l2 > 1e-4) rc = 1;
    }
    {   // 四列
        const ActQ8 A{d_q, d_d};
        Kernel::q_proj<4>(W, A, d_y, 0);
        std::vector<float> y((size_t)N * R);
        cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_abs = 0, num = 0, den = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            const double a = y[i], b = ref[i];
            max_abs = std::fmax(max_abs, std::fabs(a - b));
            num += (a - b) * (a - b);
            den += b * b;
        }
        const double l2 = std::sqrt(num / den);
        std::printf("[gpu] q_proj<4>  max_abs=%.3e  rel_l2=%.3e\n", max_abs, l2);
        if (l2 > 1e-4) rc = 1;
    }

    cudaFree(d_w); cudaFree(d_q); cudaFree(d_d); cudaFree(d_y);
    std::printf(rc ? "GPU SELFTEST FAILED\n" : "GPU SELFTEST PASS\n");
    return rc;
}
