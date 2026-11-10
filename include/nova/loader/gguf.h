// nova_v2/loader/gguf.h — 只读 GGUF 元数据 + 张量表 + 特征签名
// 加载器不认识"模型名"，只认识这套从文件里读出来的签名。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

namespace nova {

enum class GgufType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

// ggml 张量类型（V2 会遇到的；V2-Lite 的 GGUF 是混量化，必须认全）
enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14
};

// 每 32/256 个元素一块的字节数（用于算张量尺寸与行偏移）
inline constexpr uint32_t ggml_type_block_elems(GgmlType t) {
    switch (t) {
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0:
        case GgmlType::Q5_1: case GgmlType::Q8_0: return 32;
        case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K:
        case GgmlType::Q5_K: case GgmlType::Q6_K: return 256;
        default: return 1;   // F32 / F16
    }
}

inline constexpr uint32_t ggml_type_block_bytes(GgmlType t) {
    switch (t) {
        case GgmlType::F32:  return 4;
        case GgmlType::F16:  return 2;
        case GgmlType::Q4_0: return 18;    // fp16 d + 16 B qs
        case GgmlType::Q4_1: return 20;    // fp16 d,m + 16 B qs
        case GgmlType::Q5_0: return 22;    // fp16 d + 4 B qh + 16 B qs
        case GgmlType::Q5_1: return 24;    // fp16 d,m + 4 B qh + 16 B qs
        case GgmlType::Q8_0: return 34;    // fp16 d + 32 B qs
        case GgmlType::Q2_K: return 84;
        case GgmlType::Q3_K: return 110;
        case GgmlType::Q4_K: return 144;
        case GgmlType::Q5_K: return 176;
        case GgmlType::Q6_K: return 210;
        default: return 0;
    }
}

struct TensorInfo {
    std::string          name;
    GgmlType             type;
    std::vector<uint64_t> ne;      // ne[0] 是最快维 = 列
    uint64_t             offset;   // 相对 data 段
    uint64_t             bytes;
};

// 从 metadata 读出来的"特征签名"。它是加载器唯一认的东西。
struct ModelSignature {
    std::string arch;        // general.architecture
    uint64_t layers   = 0;   // {arch}.block_count
    uint64_t n_embd   = 0;   // {arch}.embedding_length
    uint64_t n_head   = 0;   // {arch}.attention.head_count
    uint64_t n_head_kv = 0;  // {arch}.attention.head_count_kv（数组形式填 0）
    uint64_t head_dim = 0;   // {arch}.attention.key_length，缺失则 n_embd/n_head
    uint64_t ffn      = 0;   // {arch}.feed_forward_length
    uint64_t vocab    = 0;   // tokenizer.ggml.tokens 的长度
    uint64_t ctx      = 0;   // {arch}.context_length
    double   rope_theta = 0;
    double   rms_eps    = 0;
    bool     tied = false;
    bool     kv_is_array = false;   // head_count_kv 写成数组 → 无法按标量核对
};

class GgufFile {
public:
    explicit GgufFile(const std::string& path);

    bool has(const std::string& key) const { return kv_.count(key) != 0; }

    uint64_t           u(const std::string& key) const;   // U32 / U64
    double             f(const std::string& key) const;   // F32 / F64
    const std::string& s(const std::string& key) const;   // STRING
    bool               b(const std::string& key) const;   // BOOL
    uint64_t           array_len(const std::string& key) const;
    const std::vector<std::string>& str_array(const std::string& key) const;
    GgufType           type_of(const std::string& key) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo*              find(const std::string& name) const;

    uint64_t data_offset() const { return data_offset_; }
    uint64_t table_offset() const { return table_off_; }   // KV 结束、张量表开始
    uint64_t file_size()   const { return file_size_; }
    const std::string& path() const { return path_; }

private:
    struct Value {
        GgufType type = GgufType::U32;
        uint64_t u = 0;
        double   f = 0;
        bool     b = false;
        std::string s;
        uint64_t arr_len = 0;
        GgufType arr_type = GgufType::U32;
    };

    std::string path_;
    std::unordered_map<std::string, Value> kv_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::vector<std::string>> str_arrays_;
    uint64_t data_offset_ = 0;
    uint64_t table_off_ = 0;
    uint64_t file_size_ = 0;
};

uint64_t ggml_type_bytes(GgmlType t, const std::vector<uint64_t>& ne);

// 纯读取：把签名从文件里抽出来，不做任何判断
ModelSignature read_signature(const GgufFile& g);

// general.name；缺失时退回文件名（不含目录与扩展名）
std::string general_name(const GgufFile& g);

} // namespace nova
