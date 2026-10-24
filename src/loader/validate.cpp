// nova_v2/src/loader/validate.cpp — 严格但包容：认签名，不认名字
#define NOMINMAX
#include <cctype>
#include "nova/loader/validate.h"
#include "nova/kernel.h"      // Model = ActiveModel（模具）
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#endif

namespace nova {
namespace {

const char* kYellow = "\x1b[33m";
const char* kReset  = "\x1b[0m";
const char* kDim    = "\x1b[2m";

// 控制台开 VT，黄色 [WARN] 才是黄的
void ansi_on() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */);
#endif
}

// 编译期模具 → 签名
ModelSignature mold_signature() {
    ModelSignature s;
    s.arch      = Model::arch_name;
    s.layers    = Model::L;
    s.n_embd    = Model::D;
    s.n_head    = Model::NH;
    s.n_head_kv = Model::NKV;
    s.head_dim  = Model::HD;
    s.ffn       = Model::FFN;
    s.vocab     = Model::V;
    s.ctx       = kCtx;
    s.rope_theta = Model::theta;
    s.rms_eps    = Model::eps;
    s.tied       = Model::tied;
    return s;
}

struct Diff { const char* field; std::string file; std::string mold; };

void diff_u(std::vector<Diff>& d, const char* f, uint64_t a, uint64_t b) {
    if (a != b) d.push_back({f, std::to_string(a), std::to_string(b)});
}
void diff_f(std::vector<Diff>& d, const char* f, double a, double b, double tol) {
    const double dv = a > b ? a - b : b - a;
    if (dv > tol * (b > 1.0 ? b : 1.0)) {
        char x[48], y[48];
        std::snprintf(x, sizeof(x), "%.9g", a);
        std::snprintf(y, sizeof(y), "%.9g", b);
        d.push_back({f, x, y});
    }
}
void diff_s(std::vector<Diff>& d, const char* f, const std::string& a, const std::string& b) {
    if (a != b) d.push_back({f, a, b});
}

// 名字归一化：小写 + 去掉量化/格式后缀，剩下的必须与白名单精确相等。
// 这样 "Qwen2.5-1.5B-Instruct-Q4_K_M" 是官方，"Qwen2.5-1.5B-Uncensored" 是变种。
std::string normalize_name(std::string s) {
    for (char& c : s) {
        c = (char)std::tolower((unsigned char)c);
        if (c == ' ' || c == '_') c = '-';
    }
    static const char* kTags[] = {"-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q6_k",
                                  "-q8_0", "-q4_0", "-f16", "-f32", "-gguf", ".gguf"};
    bool again = true;
    while (again) {
        again = false;
        for (const char* tag : kTags) {
            const size_t n = std::strlen(tag);
            if (s.size() >= n && s.compare(s.size() - n, n, tag) == 0) {
                s.resize(s.size() - n);
                again = true;
            }
        }
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '_' || s.back() == '-')) s.pop_back();
    return s;
}

bool is_official_name(const std::string& name) {
    const std::string n = normalize_name(name);
    for (const char* a : kModelAliases)
        if (n == normalize_name(a)) return true;
    return false;
}

[[noreturn]] void abort_with(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    NOVA_ABORT(buf);
}

struct Expect {
    std::string name;
    GgmlType    type;
    uint64_t    ne0;
    uint64_t    ne1;
};

std::vector<Expect> expected_tensors() {
    constexpr int L = Model::L, D = Model::D, NH = Model::NH, NKV = Model::NKV;
    constexpr int HD = Model::HD, FFN = Model::FFN, V = Model::V;
    std::vector<Expect> v;
    v.reserve((size_t)L * 12 + 4);

    v.push_back({"token_embd.weight", GgmlType::Q4_K, (uint64_t)D, (uint64_t)V});
    for (int i = 0; i < L; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        v.push_back({p + "attn_norm.weight", GgmlType::F32, (uint64_t)D, 0});
        v.push_back({p + "attn_q.weight",    GgmlType::Q4_K, (uint64_t)D, (uint64_t)(NH * HD)});
        if (Model::qkv_bias) v.push_back({p + "attn_q.bias", GgmlType::F32, (uint64_t)(NH * HD), 0});
        v.push_back({p + "attn_k.weight",    GgmlType::Q4_K, (uint64_t)D, (uint64_t)(NKV * HD)});
        if (Model::qkv_bias) v.push_back({p + "attn_k.bias", GgmlType::F32, (uint64_t)(NKV * HD), 0});
        v.push_back({p + "attn_v.weight",    GgmlType::Q4_K, (uint64_t)D, (uint64_t)(NKV * HD)});
        if (Model::qkv_bias) v.push_back({p + "attn_v.bias", GgmlType::F32, (uint64_t)(NKV * HD), 0});
        v.push_back({p + "attn_output.weight", GgmlType::Q4_K, (uint64_t)(NH * HD), (uint64_t)D});
        v.push_back({p + "ffn_norm.weight",    GgmlType::F32, (uint64_t)D, 0});
        v.push_back({p + "ffn_gate.weight",    GgmlType::Q4_K, (uint64_t)D, (uint64_t)FFN});
        v.push_back({p + "ffn_up.weight",      GgmlType::Q4_K, (uint64_t)D, (uint64_t)FFN});
        v.push_back({p + "ffn_down.weight",    GgmlType::Q4_K, (uint64_t)FFN, (uint64_t)D});
    }
    v.push_back({"output_norm.weight", GgmlType::F32, (uint64_t)D, 0});
    if (!Model::tied) v.push_back({"output.weight", GgmlType::Q4_K, (uint64_t)D, (uint64_t)V});
    return v;
}

void print_signature(const ModelSignature& s) {
    std::printf("[nova] 文件签名: arch=%s layers=%llu d=%llu heads=%llu:%llu hd=%llu ffn=%llu vocab=%llu ctx=%llu theta=%.6g eps=%.3g tied=%d\n",
                s.arch.c_str(), (unsigned long long)s.layers, (unsigned long long)s.n_embd,
                (unsigned long long)s.n_head, (unsigned long long)s.n_head_kv,
                (unsigned long long)s.head_dim, (unsigned long long)s.ffn,
                (unsigned long long)s.vocab, (unsigned long long)s.ctx,
                s.rope_theta, s.rms_eps, (int)s.tied);
}

} // namespace

Verification verify_model(const GgufFile& g) {
    ansi_on();

    const ModelSignature file = read_signature(g);
    const ModelSignature mold = mold_signature();
    const std::string    name = general_name(g);

    print_signature(file);

    // ---- 1. 签名逐字段比对 ----
    std::vector<Diff> diffs;
    diff_s(diffs, "arch", file.arch, mold.arch);
    diff_u(diffs, "layers", file.layers, mold.layers);
    diff_u(diffs, "n_embd", file.n_embd, mold.n_embd);
    diff_u(diffs, "n_head", file.n_head, mold.n_head);
    if (file.kv_is_array) diffs.push_back({"n_head_kv", "array", std::to_string(mold.n_head_kv)});
    else                  diff_u(diffs, "n_head_kv", file.n_head_kv, mold.n_head_kv);
    diff_u(diffs, "head_dim", file.head_dim, mold.head_dim);
    diff_u(diffs, "ffn", file.ffn, mold.ffn);
    diff_u(diffs, "vocab", file.vocab, mold.vocab);

    diff_f(diffs, "rope_theta", file.rope_theta, mold.rope_theta, 1e-6);
    diff_f(diffs, "rms_eps", file.rms_eps, mold.rms_eps, 1e-4);

    // tie 以张量表为准（部分 GGUF 的 tie_word_embeddings 元数据与张量表不一致）
    const bool has_out = g.find("output.weight") != nullptr;
    if (has_out == Model::tied)
        diffs.push_back({"tie", has_out ? "has output.weight" : "no output.weight",
                         Model::tied ? "tied (no output)" : "untied (has output)"});

    if (!diffs.empty()) {
        std::printf("\n[nova] 模型签名与编译期模具不一致（模具: %s）\n", kModelName);
        std::printf("  %-14s %-18s %-18s\n", "field", "file", "mold");
        for (const auto& d : diffs)
            std::printf("  %-14s %-18s %-18s\n", d.field, d.file.c_str(), d.mold.c_str());
        std::printf("\n");
        abort_with("不支持的模型特征（维度不匹配），请修改配置并重新编译引擎");
    }

    // ---- 2. 容量（不是相等关系） ----
    if (file.ctx < (uint64_t)kCtx)
        abort_with("上下文不足：文件声明 %llu < 内核预分配 %d，请重新编译引擎",
                   (unsigned long long)file.ctx, kCtx);

    // ---- 3. 变种：签名一致、名字不在白名单 ----
    const bool variant = !is_official_name(name);
    if (variant) {
        std::printf("%s[WARN]%s 检测到非官方变种模型 \"%s\"\n", kYellow, kReset, name.c_str());
        std::printf("%s       签名与模具 %s 完全一致（%s / L%llu / d%llu / heads %llu:%llu / hd%llu / ffn%llu / vocab%llu）%s\n",
                    kDim, kModelName, file.arch.c_str(),
                    (unsigned long long)file.layers, (unsigned long long)file.n_embd,
                    (unsigned long long)file.n_head, (unsigned long long)file.n_head_kv,
                    (unsigned long long)file.head_dim, (unsigned long long)file.ffn,
                    (unsigned long long)file.vocab, kReset);
        std::printf("%s       引擎将按预编译模具执行计算%s\n", kDim, kReset);
    }

    // ---- 4. 张量表 / 量化混合路径 ----
    const auto expect = expected_tensors();
    if (g.tensors().size() != expect.size())
        abort_with("张量数不符：文件 %llu，模具 %llu",
                   (unsigned long long)g.tensors().size(), (unsigned long long)expect.size());

    Verification rep{};
    rep.name    = name;
    rep.variant = variant;

    for (const auto& e : expect) {
        const TensorInfo* t = g.find(e.name);
        if (!t) abort_with("缺张量：%s", e.name.c_str());
        if (t->type != e.type) {
            if (t->type == GgmlType::Q6_K)
                abort_with("量化混合路径不符：%s 是 Q6_K（stock Q4_K_M 的 attn_v/ffn_down 就是 Q6_K），"
                           "本模具只接受纯 Q4_K，请先 requant", e.name.c_str());
            abort_with("量化类型不符：%s 是 type=%u，模具要求 type=%u",
                       e.name.c_str(), (unsigned)t->type, (unsigned)e.type);
        }
        if (t->ne.size() < (e.ne1 ? 2u : 1u)) abort_with("张量 %s 维数不足", e.name.c_str());
        if (t->ne[0] != e.ne0)
            abort_with("张量 %s 第 0 维 = %llu，模具要求 %llu", e.name.c_str(),
                       (unsigned long long)t->ne[0], (unsigned long long)e.ne0);
        if (e.ne1 && t->ne[1] != e.ne1)
            abort_with("张量 %s 第 1 维 = %llu，模具要求 %llu", e.name.c_str(),
                       (unsigned long long)t->ne[1], (unsigned long long)e.ne1);
        if (e.type == GgmlType::Q4_K) { rep.weight_bytes += t->bytes; ++rep.n_weight; }
        else ++rep.n_norm;
    }

    // ---- 5. 预算与文件完整性 ----
    if (rep.weight_bytes > kWeightBudget)
        abort_with("权重 %llu 字节超过 7 GiB 预算", (unsigned long long)rep.weight_bytes);

    uint64_t need = g.data_offset();
    for (const auto& t : g.tensors()) {
        const uint64_t end = g.data_offset() + t.offset + t.bytes;
        if (end > need) need = end;
    }
    if (g.file_size() < need)
        abort_with("文件不完整：需要 %llu 字节，实际 %llu",
                   (unsigned long long)need, (unsigned long long)g.file_size());

    return rep;
}

} // namespace nova
