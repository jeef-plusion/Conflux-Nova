#include "nova/kernel.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <cstring>
using namespace nova;

namespace nova { template <int K, int R> void launch_gemv_ref(const Q4KWeight&, const ActQ8&, float*, cudaStream_t); }

static uint16_t h2(float v) { __half h = __float2half(v); uint16_t u; std::memcpy(&u, &h, 2); return u; }

template <int K, int R>
static void one(const char* name) {
    std::mt19937 rng(7);
    std::vector<block_q4_K> w((size_t)R * (K / 256));
    for (auto& b : w) {
        b.d = h2(0.002f); b.dmin = h2(0.001f);
        for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t)(rng() & 0xFF);
        for (int i = 0; i < 128; ++i) b.qs[i] = (uint8_t)(rng() & 0xFF);
    }
    std::vector<float> x(K);
    for (auto& v : x) v = (float)((int)(rng() % 2000) - 1000) / 500.f;
    std::vector<int8_t> xq(K);
    std::vector<float> xd(K / 32);
    std::vector<int> xs(K / 16);
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f;
        const float inv = d > 0 ? 1.f / d : 0.f;
        int s0 = 0, s1 = 0;
        for (int i = 0; i < 16; ++i) { const int v = (int)std::lrint(x[b * 32 + i] * inv); xq[b * 32 + i] = (int8_t)v; s0 += v; }
        for (int i = 16; i < 32; ++i) { const int v = (int)std::lrint(x[b * 32 + i] * inv); xq[b * 32 + i] = (int8_t)v; s1 += v; }
        xd[b] = d;
        xs[b * 2] = s0; xs[b * 2 + 1] = s1;
    }
    block_q4_K* dw = nullptr; int8_t* dq = nullptr; float* dd = nullptr; int* ds = nullptr;
    float* o1 = nullptr; float* o2 = nullptr;
    cudaMalloc(&dw, w.size() * sizeof(block_q4_K));
    cudaMalloc(&dq, K); cudaMalloc(&dd, (K / 32) * 4); cudaMalloc(&ds, (K / 16) * 4);
    cudaMalloc(&o1, (size_t)R * 4); cudaMalloc(&o2, (size_t)R * 4);
    cudaMemcpy(dw, w.data(), w.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice);
    cudaMemcpy(dq, xq.data(), K, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, xd.data(), (K / 32) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, xs.data(), (K / 16) * 4, cudaMemcpyHostToDevice);
    const Q4KWeight W{(const uint8_t*)dw, R, K};
    const ActQ8 A{dq, dd, ds};
    launch_gemv<K, R>(W, A, o1, 0);
    launch_gemv_ref<K, R>(W, A, o2, 0);
    cudaDeviceSynchronize();
    std::vector<float> y1(R), y2(R);
    cudaMemcpy(y1.data(), o1, (size_t)R * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), o2, (size_t)R * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int r = 0; r < R; ++r) { num += (y1[r] - y2[r]) * (y1[r] - y2[r]); den += (double)y2[r] * y2[r]; }
    const double rel = std::sqrt(num / den);
    std::printf("%-10s K=%-6d R=%-7d rel_l2=%.3e  %s\n", name, K, R, rel, rel < 1e-5 ? "OK" : "MISMATCH");
    cudaFree(dw); cudaFree(dq); cudaFree(dd); cudaFree(ds); cudaFree(o1); cudaFree(o2);
}

int main() {
    one<Model::D, Model::NH * Model::HD>("q_proj");
    one<Model::D, Model::NKV * Model::HD>("k_proj");
    one<Model::FFN, Model::D>("ffn_down");
    one<Model::D, Model::V>("lm_head");
    return 0;
}
