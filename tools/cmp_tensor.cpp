// tools/cmp_tensor.cpp — 比较原文件 Q6_K 张量与转换后 Q4_K 张量的反量化结果
#include "nova/loader/gguf.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

using namespace nova;

struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };

static void deq6(const block_q6_K* b, float* y) {
    const float d = half_to_float(b->d);
    const uint8_t *ql = b->ql, *qh = b->qh;
    const int8_t* sc = b->scales;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l]      = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

static std::vector<char> slurp(const char* path, uint64_t off, uint64_t n) {
    std::ifstream f(path, std::ios::binary);
    f.seekg((std::streamoff)off);
    std::vector<char> b((size_t)n);
    f.read(b.data(), (std::streamsize)n);
    return b;
}

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s <orig> <new> <tensor>\n", argv[0]); return 2; }
    GgufFile a(argv[1]), b(argv[2]);
    const TensorInfo* ta = a.find(argv[3]);
    const TensorInfo* tb = b.find(argv[3]);
    if (!ta || !tb) { std::fprintf(stderr, "tensor not found\n"); return 1; }
    std::printf("orig type=%u bytes=%llu | new type=%u bytes=%llu\n", (unsigned)ta->type,
                (unsigned long long)ta->bytes, (unsigned)tb->type, (unsigned long long)tb->bytes);

    const uint64_t NB = 64;   // 抽样 64 个 block
    auto ra = slurp(argv[1], a.data_offset() + ta->offset, NB * 210);
    auto rb = slurp(argv[2], b.data_offset() + tb->offset, NB * 144);

    double num = 0, den = 0, amax = 0;
    for (uint64_t i = 0; i < NB; ++i) {
        float y[256], z[256];
        deq6((const block_q6_K*)(ra.data() + i * 210), y);
        dequant_block((const block_q4_K*)(rb.data() + i * 144), z);
        for (int k = 0; k < 256; ++k) {
            num += (z[k] - y[k]) * (z[k] - y[k]);
            den += (double)y[k] * y[k];
            if (std::fabs(y[k]) > amax) amax = std::fabs(y[k]);
        }
        if (i == 0) std::printf("first 6: q6 %.6f %.6f %.6f | q4 %.6f %.6f %.6f\n",
                                y[0], y[1], y[2], z[0], z[1], z[2]);
    }
    std::printf("rel_rms=%.5f  max|q6|=%.5f\n", std::sqrt(num / den), amax);
    return 0;
}
