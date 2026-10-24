// bench/debug_logits.cpp — 看 top-k logits，判断前向是否合理
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "nova/model.h"

using namespace nova;

int main(int argc, char** argv) {
    const char* path = argv[1];
    const int n = argc > 2 ? std::atoi(argv[2]) : 4;
    Engine e;
    e.load(path, 64);
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = 9707 + i;
    e.prefill(ids.data(), n);
    std::printf("cuda err after prefill: %s\n", cudaGetErrorString(cudaGetLastError()));

    std::vector<float> lg(Model::V);
    cudaMemcpy(lg.data(), e.logits(), sizeof(float) * Model::V, cudaMemcpyDeviceToHost);
    std::vector<int> idx(Model::V);
    for (int i = 0; i < Model::V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    std::printf("top10:");
    for (int i = 0; i < 10; ++i) std::printf(" %d(%.2f)", idx[i], lg[idx[i]]);
    std::printf("\n");
    double mn = 1e30, mx = -1e30, mean = 0;
    for (int i = 0; i < Model::V; ++i) { mn = std::min(mn, (double)lg[i]); mx = std::max(mx, (double)lg[i]); mean += lg[i]; }
    std::printf("logits min=%.2f max=%.2f mean=%.2f\n", mn, mx, mean / Model::V);
    return 0;
}
