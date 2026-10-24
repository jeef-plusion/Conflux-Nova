#include "nova/kernel.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <vector>
using namespace nova;

template <int K, int R>
static void run(const char* name) {
    block_q4_K* w = nullptr; int8_t* xq = nullptr; float* xd = nullptr; int* xs = nullptr; float* o = nullptr;
    cudaMalloc(&w, (size_t)R * (K / 256) * sizeof(block_q4_K));
    cudaMalloc(&xq, K);
    cudaMalloc(&xd, (K / 32) * 4);
    cudaMalloc(&xs, (K / 32) * 4);
    cudaMalloc(&o, (size_t)R * 4);
    cudaMemset(w, 0, (size_t)R * (K / 256) * sizeof(block_q4_K));
    cudaMemset(xq, 1, K);
    cudaMemset(xd, 0, (K / 32) * 4);
    cudaMemset(xs, 0, (K / 32) * 4);
    const Q4KWeight W{(const uint8_t*)w, R, K};
    const ActQ8 A{xq, xd, xs};
    launch_gemv<K, R>(W, A, o, 0);
    cudaDeviceSynchronize();
    std::printf("%-10s K=%-6d R=%-7d err=%s\n", name, K, R, cudaGetErrorString(cudaGetLastError()));
    cudaFree(w); cudaFree(xq); cudaFree(xd); cudaFree(xs); cudaFree(o);
}

int main() {
    run<Model::D, Model::NH * Model::HD>("q_proj");
    run<Model::D, Model::NKV * Model::HD>("k_proj");
    run<Model::NH * Model::HD, Model::D>("o_proj");
    run<Model::D, Model::FFN>("ffn_gate");
    run<Model::FFN, Model::D>("ffn_down");
    run<Model::D, Model::V>("lm_head");
    return 0;
}
