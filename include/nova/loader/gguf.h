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

// ggml 张量类型（只列 V2 会遇到的）
enum class GgmlType : uint32_t { F32 = 0, F16 = 1, Q4_K = 12, Q6_K = 14, Q8_0 = 8 };

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
