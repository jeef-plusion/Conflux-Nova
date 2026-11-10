// gguf_dump.cpp — 裸解析 GGUF 头（不依赖 nova::GgufFile 的类型白名单）
//
// 目的：nova 的 loader 只认识 F32/F16/Q4_K/Q6_K/Q8_0，遇到别的 ggml type 直接 abort
// （gguf.cpp:64 "unknown tensor type"）。要看 DeepSeek-V2-Lite 的 GGUF 到底用了哪些量化类型、
// 专家张量叫什么/什么形状，就得绕开那个白名单。
//
// 只读元数据段与张量表，不碰权重数据 —— 所以给一个被截断的文件也能跑。
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

template <class T>
bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

bool rd_str(FILE* f, std::string& s) {
    std::uint64_t n = 0;
    if (!rd(f, n)) return false;
    if (n > (1ull << 22)) return false;          // 明显不对 (4 MB)
    s.resize(static_cast<std::size_t>(n));
    if (n > 0 && std::fread(&s[0], 1, static_cast<std::size_t>(n), f) != n) return false;
    return true;
}

// 跳过（或计数）一个 value
bool skip_value(FILE* f, std::uint32_t type, std::uint64_t& bytes, int depth);

bool skip_array(FILE* f, std::uint64_t& bytes) {
    std::uint32_t atype = 0;
    std::uint64_t alen = 0;
    if (!rd(f, atype) || !rd(f, alen)) return false;
    bytes += 4 + 8;
    for (std::uint64_t i = 0; i < alen; ++i) {
        if (!skip_value(f, atype, bytes, 1)) return false;
    }
    return true;
}

bool skip_value(FILE* f, std::uint32_t type, std::uint64_t& bytes, int depth) {
    if (depth > 4) return false;
    switch (type) {
        case 0: { std::uint8_t v; if (!rd(f, v)) return false; bytes += 1; return true; }
        case 1: { std::int8_t v; if (!rd(f, v)) return false; bytes += 1; return true; }
        case 2: { std::uint16_t v; if (!rd(f, v)) return false; bytes += 2; return true; }
        case 3: { std::int16_t v; if (!rd(f, v)) return false; bytes += 2; return true; }
        case 4: { std::uint32_t v; if (!rd(f, v)) return false; bytes += 4; return true; }
        case 5: { std::int32_t v; if (!rd(f, v)) return false; bytes += 4; return true; }
        case 6: { float v; if (!rd(f, v)) return false; bytes += 4; return true; }
        case 7: { std::uint8_t v; if (!rd(f, v)) return false; bytes += 1; return true; }
        case 8: { std::string s; if (!rd_str(f, s)) return false; bytes += 8 + s.size(); return true; }
        case 9: return skip_array(f, bytes);
        case 10: { std::uint64_t v; if (!rd(f, v)) return false; bytes += 8; return true; }
        case 11: { std::int64_t v; if (!rd(f, v)) return false; bytes += 8; return true; }
        case 12: { double v; if (!rd(f, v)) return false; bytes += 8; return true; }
        default: return false;
    }
}

const char* type_name(std::uint32_t t) {
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        default: return "?";
    }
}

struct Tensor {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> ne;
};

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* path = (argc > 1) ? argv[1] : "E:\\dsv2_head.bin";
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("cannot open %s\n", path); return 1; }

    char magic[4];
    std::uint32_t ver = 0; std::uint64_t n_tensors = 0, n_kv = 0;
    std::fread(magic, 1, 4, f);
    rd(f, ver);
    rd(f, n_tensors);
    rd(f, n_kv);
    std::printf("magic=%.4s version=%u n_tensors=%llu n_kv=%llu\n", magic, ver, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    std::uint64_t bytes = 0;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!rd_str(f, key)) { std::printf("kv[%llu] key read fail\n", (unsigned long long)i); break; }
        std::uint32_t type = 0;
        if (!rd(f, type)) { std::printf("kv[%llu] type read fail\n", (unsigned long long)i); break; }
        std::uint64_t before = bytes;
        if (!skip_value(f, type, bytes, 0)) {
            std::printf("kv[%llu] %s type=%u SKIP FAIL\n", (unsigned long long)i, key.c_str(), type);
            break;
        }
        const bool interesting = (key.size() < 60) && (key.find("block_count") != std::string::npos ||
                                 key.find("n_expert") != std::string::npos ||
                                 key.find("moe") != std::string::npos ||
                                 key.find("general.name") != std::string::npos ||
                                 key.find("expert") != std::string::npos ||
                                 key.find("ffn") != std::string::npos);
        if (interesting) std::printf("kv[%llu] %-52s type=%-2u bytes=%llu\n", (unsigned long long)i, key.c_str(), type,
                                     static_cast<unsigned long long>(bytes - before));
    }

    std::printf("\n--- tensors (first 12 + all ffn/attn of blk.0/1) ---\n");
    std::map<std::uint32_t, int> type_hist;
    std::vector<Tensor> show;
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        if (!rd_str(f, t.name)) { std::printf("tensor[%llu] name fail\n", (unsigned long long)i); break; }
        std::uint32_t n_dims = 0;
        if (!rd(f, n_dims)) break;
        if (n_dims > 8) { std::printf("tensor[%llu] bad n_dims=%u\n", (unsigned long long)i, n_dims); break; }
        t.ne.resize(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            if (!rd(f, t.ne[d])) { std::printf("tensor dims fail\n"); std::fclose(f); return 0; }
        }
        std::uint64_t off = 0;
        if (!rd(f, t.type) || !rd(f, off)) break;
        type_hist[t.type]++;
        const bool want = (i < 12) || (t.type == 6) || (t.type == 8) || (t.type == 14) || (t.name.find("blk.0.") != std::string::npos) ||
                          (t.name.find("blk.1.") != std::string::npos) ||
                          (t.name.find("output") != std::string::npos);
        if (want) show.push_back(t);
    }
    for (const auto& t : show) {
        std::printf("  %-34s type=%-2u %-8s ne=[", t.name.c_str(), t.type, type_name(t.type));
        for (std::size_t d = 0; d < t.ne.size(); ++d) std::printf("%s%llu", d ? "," : "",
                                                                 static_cast<unsigned long long>(t.ne[d]));
        std::printf("]\n");
    }

    std::printf("\n--- quant type histogram (tensors parsed) ---\n");
    for (const auto& kv : type_hist) {
        std::printf("  %-8s (id %-2u) x %d\n", type_name(kv.first), kv.first, kv.second);
    }
    std::fclose(f);
    return 0;
}
