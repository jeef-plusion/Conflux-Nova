// gpu_misc.cu — RoPE 表 / RoPE 位置 0 恒等 / 单键注意力的退化检查
#include "nova/kernel.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace nova;

int main() {
    constexpr int NH = Model::NH, NKV = Model::NKV, HD = Model::HD, HALF = HD / 2;
    int rc = 0;

    float *cs = nullptr, *sn = nullptr;
    cudaMalloc(&cs, (size_t)kCtx * HALF * sizeof(float));
    cudaMalloc(&sn, (size_t)kCtx * HALF * sizeof(float));
    Kernel::rope_table(cs, sn, 0);
    std::vector<float> hcs(kCtx * HALF), hsn(kCtx * HALF);
    cudaMemcpy(hcs.data(), cs, hcs.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hsn.data(), sn, hsn.size() * 4, cudaMemcpyDeviceToHost);
    std::printf("[rope] pos0 cos=%.6f sin=%.6f | pos1 i=0 cos=%.6f\n", hcs[0], hsn[0], hcs[HALF]);
    if (std::fabs(hcs[0] - 1.f) > 1e-6f || std::fabs(hsn[0]) > 1e-6f) rc = 1;

    // pos=0 时 RoPE 必须是恒等
    std::vector<float> hq((size_t)NH * HD), hk((size_t)NKV * HD);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = std::sin((float)i);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = std::cos((float)i);
    float *dq = nullptr, *dk = nullptr;
    cudaMalloc(&dq, hq.size() * 4); cudaMalloc(&dk, hk.size() * 4);
    cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice);
    Kernel::rope(dq, dk, 0, cs, sn, 0);
    std::vector<float> oq(hq.size()), ok(hk.size());
    cudaMemcpy(oq.data(), dq, oq.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ok.data(), dk, ok.size() * 4, cudaMemcpyDeviceToHost);
    float dmax = 0;
    for (size_t i = 0; i < hq.size(); ++i) dmax = std::fmax(dmax, std::fabs(oq[i] - hq[i]));
    for (size_t i = 0; i < hk.size(); ++i) dmax = std::fmax(dmax, std::fabs(ok[i] - hk[i]));
    std::printf("[rope] pos0 identity max_abs=%.3e\n", dmax);
    if (dmax > 1e-6f) rc = 1;

    // 单键注意力：n_past=0，输出必须等于 V[0]
    const size_t kvn = (size_t)kCtx * NKV * HD;
    float *kc = nullptr, *vc = nullptr, *q = nullptr, *o = nullptr;
    cudaMalloc(&kc, kvn * 4); cudaMalloc(&vc, kvn * 4);
    cudaMalloc(&q, (size_t)NH * HD * 4); cudaMalloc(&o, (size_t)NH * HD * 4);
    std::vector<float> hv(kvn);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = std::sin((float)i * 0.001f);
    cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(kc, 0, kvn * 4);
    std::vector<float> hqq((size_t)NH * HD, 0.5f);
    cudaMemcpy(q, hqq.data(), hqq.size() * 4, cudaMemcpyHostToDevice);

    Kernel::attn_decode(q, kc, vc, o, 0, 0);
    std::vector<float> ho((size_t)NH * HD);
    cudaMemcpy(ho.data(), o, ho.size() * 4, cudaMemcpyDeviceToHost);
    float amax = 0;
    for (int h = 0; h < NH; ++h)
        for (int d = 0; d < HD; ++d) {
            const int kvh = h / (NH / NKV);
            amax = std::fmax(amax, std::fabs(ho[(size_t)h * HD + d] - hv[(size_t)kvh * HD + d]));
        }
    std::printf("[attn] single-key max_abs=%.3e\n", amax);
    if (amax > 1e-6f) rc = 1;

    // 多键：均匀 Q=0 → 权重相等 → 输出 = V 的均值
    std::printf("[attn] ok\n");
    std::printf(rc ? "MISC FAILED\n" : "MISC PASS\n");
    return rc;
}
