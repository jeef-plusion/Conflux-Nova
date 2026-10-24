// tools/requant_q4k.cpp — 把 GGUF 里的 Q6_K 张量离线压成 Q4_K，输出纯 Q4_K 模具文件。
// 只改张量类型/偏移，KV 段按字节原样拷贝。
#include "nova/loader/gguf.h"
#include "nova/quant/q4_k.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <cerrno>

using namespace nova;

namespace {

struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(block_q6_K) == 210, "q6_K block must be 210 bytes");

uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;
    if (e == 0) return (uint16_t)sign;
    const int exp = (int)e - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        const uint32_t m = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | (m >> 13));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

void dequant_q6k(const block_q6_K* b, float* y) {
    const float d = half_to_float(b->d);
    const uint8_t* ql = b->ql;
    const uint8_t* qh = b->qh;
    const int8_t*  sc = b->scales;
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

void quantize_q4k(const float* x, block_q4_K* y) {
    uint8_t L[256];
    float scales[8], mins[8];

    for (int j = 0; j < 8; ++j) {
        float mx = -1e30f, mn = 1e30f;
        for (int l = 0; l < 32; ++l) {
            const float v = x[32 * j + l];
            mx = v > mx ? v : mx;
            mn = v < mn ? v : mn;
        }
        if (mn > 0.f) mn = 0.f;
        float d = (mx - mn) / 15.f;
        if (d <= 0.f) d = 1e-8f;
        scales[j] = d;
        mins[j]   = -mn;
    }

    float max_scale = 0.f, max_min = 0.f;
    for (int j = 0; j < 8; ++j) {
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j]   > max_min)   max_min   = mins[j];
    }
    const float inv_scale = max_scale > 0.f ? 63.f / max_scale : 0.f;
    const float inv_min   = max_min   > 0.f ? 63.f / max_min   : 0.f;

    std::memset(y->scales, 0, 12);
    for (int j = 0; j < 8; ++j) {
        int ls = (int)std::lround(inv_scale * scales[j]);
        int lm = (int)std::lround(inv_min * mins[j]);
        if (ls > 63) ls = 63;
        if (lm > 63) lm = 63;
        if (j < 4) {
            y->scales[j]     = (uint8_t)ls;
            y->scales[j + 4] = (uint8_t)lm;
        } else {
            y->scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            y->scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
            y->scales[j]     |= (uint8_t)((lm >> 4) << 6);
        }
    }
    y->d    = f2h(max_scale / 63.f);
    y->dmin = f2h(max_min / 63.f);

    const float dg  = half_to_float(y->d);
    const float dmg = half_to_float(y->dmin);
    for (int j = 0; j < 8; ++j) {
        uint8_t sc, m;
        get_scale_min_k4(j, y->scales, sc, m);
        const float d  = dg * sc;
        const float dm = dmg * m;
        for (int i = 0; i < 32; ++i) {
            int v = 0;
            if (d > 0.f) v = (int)std::lround((x[32 * j + i] + dm) / d);
            if (v < 0) v = 0;
            if (v > 15) v = 15;
            L[32 * j + i] = (uint8_t)v;
        }
    }
    for (int j = 0; j < 4; ++j)
        for (int l = 0; l < 32; ++l)
            y->qs[32 * j + l] = (uint8_t)(L[64 * j + l] | (L[64 * j + 32 + l] << 4));
}

} // namespace

static int selftest() {
    // 随机 256 个值 -> Q4_K -> 反量化，看误差量级
    float x[256], y[256];
    double worst = 0;
    for (int it = 0; it < 64; ++it) {
        for (int i = 0; i < 256; ++i) x[i] = (float)((it * 37 + i * 13) % 1000 - 500) / 250.f;
        block_q4_K b;
        quantize_q4k(x, &b);
        dequant_block(&b, y);
        double num = 0, den = 0;
        for (int i = 0; i < 256; ++i) { const double d = y[i] - x[i]; num += d * d; den += (double)x[i] * x[i]; }
        const double rel = std::sqrt(num / den);
        if (rel > worst) worst = rel;
    }
    std::printf("[selftest] q4k encode/decode worst rel_rms = %.4f\n", worst);
    return worst < 0.08 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc != 3) { std::fprintf(stderr, "usage: %s <in.gguf> <out.gguf>\n", argv[0]); return 2; }

    GgufFile g(argv[1]);
    const uint64_t table_off = g.table_offset();
    const uint64_t data_off  = g.data_offset();
    const uint64_t align     = g.has("general.alignment") ? g.u("general.alignment") : 32;

    // 新偏移表
    struct Entry { const TensorInfo* t; uint32_t type; uint64_t off, bytes; };
    std::vector<Entry> es;
    uint64_t cur = 0;
    uint64_t n_q6 = 0, n_q4 = 0;
    for (const auto& t : g.tensors()) {
        Entry e{&t, (uint32_t)t.type, 0, t.bytes};
        if (t.type == GgmlType::Q6_K) {
            uint64_t n = 1;
            for (uint64_t d : t.ne) n *= d;
            e.type  = (uint32_t)GgmlType::Q4_K;
            e.bytes = (n / 256) * 144;
            ++n_q6;
        } else if (t.type == GgmlType::Q4_K) ++n_q4;
        e.off = cur;
        cur = (cur + e.bytes + align - 1) / align * align;
        if (es.size() < 12)
            std::printf("[layout] %-32s type=%u ne0=%llu ne1=%llu bytes=%llu off=%llu\n",
                        t.name.c_str(), e.type, (unsigned long long)(t.ne.size() > 0 ? t.ne[0] : 0),
                        (unsigned long long)(t.ne.size() > 1 ? t.ne[1] : 0),
                        (unsigned long long)e.bytes, (unsigned long long)e.off);
        es.push_back(e);
    }
    const uint64_t total_data = cur;

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::fprintf(stderr, "open input failed: %s (errno=%d)\n", argv[1], errno); return 1; }
    std::ofstream out(argv[2], std::ios::binary);
    if (!out) { std::fprintf(stderr, "open output failed: %s (errno=%d)\n", argv[2], errno); return 1; }

    // 1) header + KV 原样拷贝
    std::vector<char> buf((size_t)table_off);
    in.read(buf.data(), (std::streamsize)table_off);
    out.write(buf.data(), (std::streamsize)table_off);

    // 2) 新张量表
    for (const auto& e : es) {
        const TensorInfo& t = *e.t;
        const uint64_t n = t.name.size();
        out.write((const char*)&n, 8);
        out.write(t.name.data(), (std::streamsize)n);
        const uint32_t nd = (uint32_t)t.ne.size();
        out.write((const char*)&nd, 4);
        for (uint64_t d : t.ne) out.write((const char*)&d, 8);
        out.write((const char*)&e.type, 4);
        out.write((const char*)&e.off, 8);
    }

    // 3) 对齐填充
    const uint64_t pos = (uint64_t)out.tellp();
    const uint64_t pad = (pos + align - 1) / align * align - pos;
    std::vector<char> zeros((size_t)pad, 0);
    out.write(zeros.data(), (std::streamsize)pad);
    const uint64_t data_start = (uint64_t)out.tellp();

    // 4) 数据段
    std::printf("[requant] %zu tensors: %llu Q4_K kept, %llu Q6_K -> Q4_K\n",
                g.tensors().size(), (unsigned long long)n_q4, (unsigned long long)n_q6);

    std::vector<char>  ibuf(1 << 22);
    std::vector<char>  obuf(1 << 22);
    double max_rel_seen = 0.0;
    for (const auto& e : es) {
        const TensorInfo& t = *e.t;
        // 每个张量按表里的偏移落位（张量之间有对齐填充）
        const uint64_t want = data_start + e.off;
        const uint64_t have = (uint64_t)out.tellp();
        if (want > have) {
            std::vector<char> z((size_t)(want - have), 0);
            out.write(z.data(), (std::streamsize)z.size());
        } else if (want < have) {
            std::fprintf(stderr, "layout drift at %s: want=%llu have=%llu (off=%llu bytes=%llu)\n",
                         t.name.c_str(), (unsigned long long)want, (unsigned long long)have,
                         (unsigned long long)e.off, (unsigned long long)e.bytes);
            return 1;
        }
        in.seekg((std::streamoff)(data_off + t.offset), std::ios::beg);

        if (t.type == GgmlType::Q6_K) {
            const uint64_t nb = t.bytes / 210;
            uint64_t done = 0;
            while (done < nb) {
                const uint64_t n = (nb - done) > 2048 ? 2048 : (nb - done);
                in.read(ibuf.data(), (std::streamsize)(n * 210));
                const block_q6_K* src = (const block_q6_K*)ibuf.data();
                block_q4_K* dst = (block_q4_K*)obuf.data();
                for (uint64_t i = 0; i < n; ++i) {
                    float y[256];
                    dequant_q6k(src + i, y);
                    quantize_q4k(y, dst + i);
                    if (i % 64 == 0) {   // 抽样误差
                        float back[256];
                        dequant_block(dst + i, back);
                        double num = 0, den = 0;
                        for (int k = 0; k < 256; ++k) { const double d = back[k] - y[k]; num += d * d; den += (double)y[k] * y[k]; }
                        const double rel = std::sqrt(num / (den + 1e-30));
                        if (rel > max_rel_seen) max_rel_seen = rel;
                    }
                }
                out.write(obuf.data(), (std::streamsize)(n * 144));
                done += n;
            }
        } else {
            uint64_t left = t.bytes;
            while (left) {
                const size_t n = (size_t)(left > ibuf.size() ? ibuf.size() : left);
                in.read(ibuf.data(), (std::streamsize)n);
                out.write(ibuf.data(), (std::streamsize)n);
                left -= n;
            }
        }
    }

    // 5) 尾部对齐
    const uint64_t end = (uint64_t)out.tellp();
    const uint64_t tpad = (end + align - 1) / align * align - end;
    std::vector<char> tz((size_t)tpad, 0);
    out.write(tz.data(), (std::streamsize)tpad);
    out.close();

    std::printf("[requant] done: data=%.3f GiB  sampled q6->q4 rel_rms=%.4f\n",
                total_data / 1073741824.0, max_rel_seen);
    return 0;
}
