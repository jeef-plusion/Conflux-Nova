// nova_v2/src/loader/gguf.cpp — GGUF v3 顺序解析；出错一律 NOVA_ABORT
#include "nova/loader/gguf.h"
#include "nova/config.h"
#include <cstring>
#include <limits>

namespace nova {
namespace {

template <class T>
T rd(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) NOVA_ABORT("gguf truncated");
    return v;
}

std::string rd_str(std::ifstream& f) {
    const uint64_t n = rd<uint64_t>(f);
    if (n > (1ull << 30)) NOVA_ABORT("gguf string too long");
    std::string s(n, '\0');
    f.read(s.data(), (std::streamsize)n);
    if (!f) NOVA_ABORT("gguf truncated (string)");
    return s;
}

void skip_value(std::ifstream& f, GgufType t);

void skip_array(std::ifstream& f) {
    const uint32_t et = rd<uint32_t>(f);
    const uint64_t n  = rd<uint64_t>(f);
    if (n > (1ull << 32)) NOVA_ABORT("gguf array too long");
    for (uint64_t i = 0; i < n; ++i) skip_value(f, (GgufType)et);
}

void skip_value(std::ifstream& f, GgufType t) {
    switch (t) {
        case GgufType::U8:  case GgufType::I8:  case GgufType::BOOL: f.seekg(1, std::ios::cur); break;
        case GgufType::U16: case GgufType::I16: f.seekg(2, std::ios::cur); break;
        case GgufType::U32: case GgufType::I32: case GgufType::F32: f.seekg(4, std::ios::cur); break;
        case GgufType::U64: case GgufType::I64: case GgufType::F64: f.seekg(8, std::ios::cur); break;
        case GgufType::STRING: {
            const uint64_t n = rd<uint64_t>(f);
            f.seekg((std::streamoff)n, std::ios::cur);
            break;
        }
        case GgufType::ARRAY: skip_array(f); return;
    }
    if (!f) NOVA_ABORT("gguf truncated (skip)");
}

} // namespace

uint64_t ggml_type_bytes(GgmlType t, const std::vector<uint64_t>& ne) {
    uint64_t n = 1;
    for (uint64_t d : ne) n *= d;
    switch (t) {
        case GgmlType::F32:  return n * 4;
        case GgmlType::F16:  return n * 2;
        case GgmlType::Q8_0: return (n / 32) * 34;
        case GgmlType::Q4_K: return (n / 256) * 144;
        case GgmlType::Q6_K: return (n / 256) * 210;
    }
    NOVA_ABORT("unknown tensor type");
    return 0;
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) NOVA_ABORT(("cannot open: " + path).c_str());

    f.seekg(0, std::ios::end);
    file_size_ = (uint64_t)f.tellg();
    f.seekg(0, std::ios::beg);

    uint32_t magic = rd<uint32_t>(f);
    if (magic != 0x46554747u) NOVA_ABORT("not a GGUF file (bad magic)");
    const uint32_t version = rd<uint32_t>(f);
    if (version != 3) NOVA_ABORT("only GGUF v3 is supported");

    const uint64_t n_tensors = rd<uint64_t>(f);
    const uint64_t n_kv      = rd<uint64_t>(f);

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = rd_str(f);
        const uint32_t vt = rd<uint32_t>(f);
        Value v;
        v.type = (GgufType)vt;
        switch (v.type) {
            case GgufType::U8:  v.u = rd<uint8_t>(f);  break;
            case GgufType::I8:  v.u = (uint64_t)rd<int8_t>(f); break;
            case GgufType::U16: v.u = rd<uint16_t>(f); break;
            case GgufType::I16: v.u = (uint64_t)rd<int16_t>(f); break;
            case GgufType::U32: v.u = rd<uint32_t>(f); break;
            case GgufType::I32: v.u = (uint64_t)rd<int32_t>(f); break;
            case GgufType::U64: v.u = rd<uint64_t>(f); break;
            case GgufType::I64: v.u = (uint64_t)rd<int64_t>(f); break;
            case GgufType::F32: v.f = (double)rd<float>(f); break;
            case GgufType::F64: v.f = rd<double>(f); break;
            case GgufType::BOOL: v.b = rd<uint8_t>(f) != 0; break;
            case GgufType::STRING: v.s = rd_str(f); break;
            case GgufType::ARRAY: {
                v.arr_type = (GgufType)rd<uint32_t>(f);
                v.arr_len  = rd<uint64_t>(f);
                if (v.arr_type == GgufType::STRING) {
                    std::vector<std::string> arr;
                    arr.reserve((size_t)v.arr_len);
                    for (uint64_t k = 0; k < v.arr_len; ++k) arr.push_back(rd_str(f));
                    str_arrays_[key] = std::move(arr);
                } else {
                    for (uint64_t k = 0; k < v.arr_len; ++k) skip_value(f, v.arr_type);
                }
                break;
            }
        }
        kv_[key] = std::move(v);
    }

    table_off_ = (uint64_t)f.tellg();
    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = rd_str(f);
        const uint32_t nd = rd<uint32_t>(f);
        if (nd == 0 || nd > 4) NOVA_ABORT("bad tensor rank");
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.ne[d] = rd<uint64_t>(f);
        t.type   = (GgmlType)rd<uint32_t>(f);
        t.offset = rd<uint64_t>(f);
        t.bytes  = ggml_type_bytes(t.type, t.ne);
        tensors_.push_back(std::move(t));
    }

    uint64_t align = 32;
    auto it = kv_.find("general.alignment");
    if (it != kv_.end()) align = it->second.u;

    const uint64_t pos = (uint64_t)f.tellg();
    data_offset_ = (pos + align - 1) / align * align;
}

const TensorInfo* GgufFile::find(const std::string& name) const {
    for (const auto& t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

uint64_t GgufFile::u(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) NOVA_ABORT(("missing metadata: " + key).c_str());
    return it->second.u;
}

double GgufFile::f(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) NOVA_ABORT(("missing metadata: " + key).c_str());
    return it->second.f;
}

const std::string& GgufFile::s(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) NOVA_ABORT(("missing metadata: " + key).c_str());
    return it->second.s;
}

bool GgufFile::b(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return false;
    return it->second.b;
}

uint64_t GgufFile::array_len(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) NOVA_ABORT(("missing metadata: " + key).c_str());
    return it->second.arr_len;
}

const std::vector<std::string>& GgufFile::str_array(const std::string& key) const {
    auto it = str_arrays_.find(key);
    if (it == str_arrays_.end()) NOVA_ABORT(("missing string array: " + key).c_str());
    return it->second;
}

GgufType GgufFile::type_of(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) NOVA_ABORT(("missing metadata: " + key).c_str());
    return it->second.type;
}

// ---- 特征签名：只读文件，不做判断 ----
ModelSignature read_signature(const GgufFile& g) {
    ModelSignature s;
    s.arch = g.s("general.architecture");
    const std::string p = s.arch + ".";

    s.layers  = g.u(p + "block_count");
    s.n_embd  = g.u(p + "embedding_length");
    s.n_head  = g.u(p + "attention.head_count");
    s.ffn     = g.u(p + "feed_forward_length");
    s.ctx     = g.u(p + "context_length");

    const std::string kk = p + "attention.head_count_kv";
    s.kv_is_array = (g.type_of(kk) == GgufType::ARRAY);
    s.n_head_kv   = s.kv_is_array ? 0 : g.u(kk);

    s.rope_theta = g.f(p + "rope.freq_base");
    s.rms_eps    = g.f(p + "attention.layer_norm_rms_epsilon");

    const std::string hd = p + "attention.key_length";
    s.head_dim = g.has(hd) ? g.u(hd) : (s.n_head ? s.n_embd / s.n_head : 0);

    s.vocab = g.array_len("tokenizer.ggml.tokens");
    s.tied  = g.b(p + "tie_word_embeddings");
    return s;
}

std::string general_name(const GgufFile& g) {
    if (g.has("general.name") && g.type_of("general.name") == GgufType::STRING)
        return g.s("general.name");

    std::string p = g.path();
    const size_t slash = p.find_last_of("/\\");
    if (slash != std::string::npos) p = p.substr(slash + 1);
    const size_t dot = p.find_last_of('.');
    if (dot != std::string::npos && dot > 0) p = p.substr(0, dot);
    return p;
}

} // namespace nova
