// nova_v2/code/model/sycl/runner.cpp — SYCL 端到端 decode 的实现
//
// 一层的算子序列与 CUDA 版 code/graph/qwen2.h + dense.h 逐条对应：
//   rmsnorm_quant(attn_norm) -> q/k/v gemv -> qkv bias -> rope -> kv_store -> attn_decode
//   -> quant(attn_out) -> o gemv -> x += o            (residual)
//   -> rmsnorm_quant(ffn_norm) -> gate/up gemv -> swiglu_quant -> down gemv -> x += down
// 然后 head：rms_norm(out_norm) -> quant -> lm_head gemv -> argmax/top-p。
#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "nova/config.h"
#include "nova/quant/q4_k.h"
#include "model/sycl/scope_sycl.h"     // NovaScope：NOVA_SCOPE=1 时记录每个算子的 device 时间

namespace nova {
namespace sycls {

using namespace syclk;

namespace {

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// 读一个 GGUF 张量的原始字节（Q4_K 或 F32）
std::vector<std::uint8_t> read_tensor(const GgufFile& g, const TensorInfo& ti) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(ti.bytes));
    FILE* f = std::fopen(g.path().c_str(), "rb");
    if (f == nullptr) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: 打不开 %s", g.path().c_str());
        NOVA_ABORT(m);
    }
    _fseeki64(f, static_cast<long long>(g.data_offset() + ti.offset), SEEK_SET);
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: 读张量 %s 失败（%zu/%zu 字节）", ti.name.c_str(), got, buf.size());
        NOVA_ABORT(m);
    }
    return buf;
}

} // namespace

E2ERunner::E2ERunner(const std::string& gguf_path, int ctx_max) : q_(dev_.q()) {
    if (ctx_max <= 0 || ctx_max > nova::kCtx) {
        char m[192];
        std::snprintf(m, sizeof(m), "runner: ctx_max=%d 超范围 (0, %d]", ctx_max, nova::kCtx);
        NOVA_ABORT(m);
    }
    ctx_max_ = ctx_max;

    GgufFile g(gguf_path);
    std::printf("[runner] %s\n", gguf_path.c_str());
    std::printf("[runner] 形状: L=%d D=%d NH=%d NKV=%d HD=%d FFN=%d V=%d ctx=%d tied=%d\n",
                kL, kD, kNH, kNKV, kHD, kFFN, kV, ctx_max_, static_cast<int>(Model::tied));

    // ---- 权重 ----
    const auto t0 = Clock::now();
    load_weights(g);
    std::printf("[runner] 权重加载完成: %.1f MiB，耗时 %.2f s\n",
                static_cast<double>(weight_bytes_) / (1024.0 * 1024.0), secs(t0, Clock::now()));

    // ---- 激活 / IO ----
    const std::uint64_t f32 = 4;
    x_     = static_cast<float*>(dev_.alloc(std::uint64_t(kD) * f32,          syclb::MemTag::Act, "e2e.x"));
    h_     = static_cast<float*>(dev_.alloc(std::uint64_t(kD) * f32,          syclb::MemTag::Act, "e2e.h"));
    qb_    = static_cast<float*>(dev_.alloc(std::uint64_t(kQKV) * f32,        syclb::MemTag::Act, "e2e.q"));
    kb_    = static_cast<float*>(dev_.alloc(std::uint64_t(kKV) * f32,         syclb::MemTag::Act, "e2e.k"));
    vb_    = static_cast<float*>(dev_.alloc(std::uint64_t(kKV) * f32,         syclb::MemTag::Act, "e2e.v"));
    ab_    = static_cast<float*>(dev_.alloc(std::uint64_t(kQKV) * f32,        syclb::MemTag::Act, "e2e.ao"));
    gb_    = static_cast<float*>(dev_.alloc(std::uint64_t(kFFN) * f32,        syclb::MemTag::Act, "e2e.g"));
    ub_    = static_cast<float*>(dev_.alloc(std::uint64_t(kFFN) * f32,        syclb::MemTag::Act, "e2e.u"));
    db_    = static_cast<float*>(dev_.alloc(std::uint64_t(kD) * f32,          syclb::MemTag::Act, "e2e.d"));
    logits_ = static_cast<float*>(dev_.alloc(std::uint64_t(kV) * f32,         syclb::MemTag::Io,  "e2e.logits"));
    ids_   = static_cast<int*>(dev_.alloc(std::uint64_t(4) * sizeof(int),     syclb::MemTag::Io,  "e2e.ids"));
    cs_    = static_cast<float*>(dev_.alloc(std::uint64_t(nova::kCtx) * (kHD / 2) * f32, syclb::MemTag::Act, "e2e.rope_cs"));
    sn_    = static_cast<float*>(dev_.alloc(std::uint64_t(nova::kCtx) * (kHD / 2) * f32, syclb::MemTag::Act, "e2e.rope_sn"));
    scratch_ = static_cast<float*>(dev_.alloc(std::uint64_t(attn_decode_scratch_floats()) * f32,
                                              syclb::MemTag::Scratch, "e2e.attn_scratch"));

    // ActQ8（D 宽 / FFN 宽）
    aq_  = static_cast<std::int8_t*>(dev_.alloc(std::uint64_t(kD) * 1,        syclb::MemTag::Act, "e2e.aq"));
    ad_  = static_cast<float*>(dev_.alloc(std::uint64_t(kD / 32) * f32,       syclb::MemTag::Act, "e2e.ad"));
    as_  = static_cast<std::int32_t*>(dev_.alloc(std::uint64_t(kD / 16) * f32, syclb::MemTag::Act, "e2e.as"));
    a2q_ = static_cast<std::int8_t*>(dev_.alloc(std::uint64_t(kFFN) * 1,      syclb::MemTag::Act, "e2e.a2q"));
    a2d_ = static_cast<float*>(dev_.alloc(std::uint64_t(kFFN / 32) * f32,     syclb::MemTag::Act, "e2e.a2d"));
    a2s_ = static_cast<std::int32_t*>(dev_.alloc(std::uint64_t(kFFN / 16) * f32, syclb::MemTag::Act, "e2e.a2s"));
    A_  = syclb::ActQ8View{aq_,  ad_,  as_};
    A2_ = syclb::ActQ8View{a2q_, a2d_, a2s_};

    // KV：[L][ctx][NKV*HD]，k/v 分开（与 attn_decode 的下标一致）
    const std::uint64_t kv_n = std::uint64_t(kL) * std::uint64_t(ctx_max_) * std::uint64_t(kKV);
    kvk_ = static_cast<float*>(dev_.alloc(kv_n * f32, syclb::MemTag::Kv, "e2e.kv_k"));
    kvv_ = static_cast<float*>(dev_.alloc(kv_n * f32, syclb::MemTag::Kv, "e2e.kv_v"));
    dev_.zero(kvk_, kv_n * f32);
    dev_.zero(kvv_, kv_n * f32);

    h_logits_.resize(static_cast<std::size_t>(kV));

    rope_table(cs_, sn_, q_);
    dev_.sync();
}

E2ERunner::~E2ERunner() {
    // 进程退出时由 Device 的记账负责报告；这里只把 ChunkedWeight 还回去
    for (auto& l : lw_) {
        l.q.destroy(dev_); l.k.destroy(dev_); l.v.destroy(dev_); l.o.destroy(dev_);
        l.gate.destroy(dev_); l.up.destroy(dev_); l.down.destroy(dev_);
    }
    emb_.destroy(dev_);
    outw_.destroy(dev_);
}

void E2ERunner::load_q4k(syclb::ChunkedWeight& w, const GgufFile& g, const std::string& name, int rows, int cols) {
    const TensorInfo* ti = g.find(name);
    if (ti == nullptr) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: GGUF 里没有张量 %s", name.c_str());
        NOVA_ABORT(m);
    }
    if (ti->type != GgmlType::Q4_K) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: 张量 %s 不是 Q4_K（type=%u）", name.c_str(),
                      static_cast<unsigned>(ti->type));
        NOVA_ABORT(m);
    }
    const std::int64_t row_bytes = static_cast<std::int64_t>(cols / kQK) * kBlockBytes;
    if (static_cast<std::int64_t>(ti->bytes) != row_bytes * rows) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: 张量 %s 尺寸不符（bytes=%llu，期望 %lld）", name.c_str(),
                      static_cast<unsigned long long>(ti->bytes),
                      static_cast<long long>(row_bytes * rows));
        NOVA_ABORT(m);
    }
    std::vector<std::uint8_t> host = read_tensor(g, *ti);
    w.create(dev_, rows, row_bytes, name.c_str());
    w.upload(dev_, host.data());
    weight_bytes_ += static_cast<std::int64_t>(host.size());
}

void E2ERunner::load_f32(float*& dst, const GgufFile& g, const std::string& name, int n) {
    const TensorInfo* ti = g.find(name);
    if (ti == nullptr) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: GGUF 里没有张量 %s", name.c_str());
        NOVA_ABORT(m);
    }
    if (ti->type != GgmlType::F32 || static_cast<int>(ti->bytes) != n * 4) {
        char m[256];
        std::snprintf(m, sizeof(m), "runner: 张量 %s 不是 %d 个 F32（type=%u bytes=%llu）", name.c_str(), n,
                      static_cast<unsigned>(ti->type), static_cast<unsigned long long>(ti->bytes));
        NOVA_ABORT(m);
    }
    std::vector<std::uint8_t> host = read_tensor(g, *ti);
    dst = static_cast<float*>(dev_.alloc(static_cast<std::uint64_t>(n) * 4, syclb::MemTag::Weights,
                                         name.c_str()));
    dev_.upload(dst, host.data(), static_cast<std::uint64_t>(n) * 4);
    weight_bytes_ += static_cast<std::int64_t>(n) * 4;
}

void E2ERunner::load_weights(const GgufFile& g) {
    load_q4k(emb_, g, "token_embd.weight", kV, kD);

    // 非 tie 的模型有独立的 output.weight；1.5B 是 tie（张量表里没有 output.weight）
    if (g.find("output.weight") != nullptr) {
        load_q4k(outw_, g, "output.weight", kV, kD);
        tied_ = false;
    } else {
        tied_ = true;
    }
    load_f32(out_norm_, g, "output_norm.weight", kD);

    lw_.resize(static_cast<std::size_t>(kL));
    for (int l = 0; l < kL; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        LayerW& w = lw_[static_cast<std::size_t>(l)];
        load_q4k(w.q,    g, p + "attn_q.weight",      kQKV, kD);
        load_q4k(w.k,    g, p + "attn_k.weight",      kKV,  kD);
        load_q4k(w.v,    g, p + "attn_v.weight",      kKV,  kD);
        load_q4k(w.o,    g, p + "attn_output.weight", kD,   kQKV);
        load_q4k(w.gate, g, p + "ffn_gate.weight",    kFFN, kD);
        load_q4k(w.up,   g, p + "ffn_up.weight",      kFFN, kD);
        load_q4k(w.down, g, p + "ffn_down.weight",    kD,   kFFN);
        load_f32(w.attn_norm, g, p + "attn_norm.weight", kD);
        load_f32(w.ffn_norm,  g, p + "ffn_norm.weight",  kD);
        if (Model::qkv_bias) {
            load_f32(w.qb, g, p + "attn_q.bias", kQKV);
            load_f32(w.kb, g, p + "attn_k.bias", kKV);
            load_f32(w.vb, g, p + "attn_v.bias", kKV);
        }
    }
}

void E2ERunner::check_ctx(int pos) const {
    if (pos < 0 || pos >= ctx_max_) {
        char m[192];
        std::snprintf(m, sizeof(m), "runner: pos=%d 超出 ctx_max=%d（KV 会越界）", pos, ctx_max_);
        NOVA_ABORT(m);
    }
}

void E2ERunner::embed(int t) {
    ::nova::scope::Scope::get().set_layer(-1);
    dev_.upload(ids_, &t, sizeof(int));          // H2D 4 B（NovaScope 会在 Device::upload 里记账）
    NOVA_SCOPED("embed_rows", static_cast<std::int64_t>(kD) * 4, embed_rows(x_, emb_.base(), ids_, 1, q_));
}

// 一层的 12 个算子，顺序 = code/graph/qwen2.h 的 OpList
void E2ERunner::layer(int l, int pos0) {
    const LayerW& w = lw_[static_cast<std::size_t>(l)];
    const std::int64_t kv_off = static_cast<std::int64_t>(l) * ctx_max_ * kKV;
    ::nova::scope::Scope::get().set_layer(l);

    // 每个 op 都包在 NOVA_SCOPED 里：关掉 NovaScope 时它只是 `__VA_ARGS__;`（零开销），
    // 打开时记录 name/bytes 并提交一个队列边界事件。bytes 用 ChunkedWeight::bytes()，
    // 也就是这个算子真正要读的权重字节数 —— 带宽就是这么算出来的。
    const std::int64_t bq = static_cast<std::int64_t>(w.q.bytes());
    const std::int64_t bk = static_cast<std::int64_t>(w.k.bytes());
    const std::int64_t bv = static_cast<std::int64_t>(w.v.bytes());
    const std::int64_t bo = static_cast<std::int64_t>(w.o.bytes());
    const std::int64_t bg = static_cast<std::int64_t>(w.gate.bytes());
    const std::int64_t bu = static_cast<std::int64_t>(w.up.bytes());
    const std::int64_t bd = static_cast<std::int64_t>(w.down.bytes());

    // 1) rms_norm(x, attn_norm) + quant  -> A
    NOVA_SCOPED("rmsnorm_quant", static_cast<std::int64_t>(2 * kD * 4), rmsnorm_quant(h_, A_, x_, w.attn_norm, 1, q_));

    // 2) q / k / v 三投影（Q4_K x int8 GEMV）
    NOVA_SCOPED("gemv.q", bq, gemv<kD, true>(w.q, A_, qb_, q_));
    NOVA_SCOPED("gemv.k", bk, gemv<kD, true>(w.k, A_, kb_, q_));
    NOVA_SCOPED("gemv.v", bv, gemv<kD, true>(w.v, A_, vb_, q_));

    // 3) bias（Qwen2 系有）
    if (Model::qkv_bias) {
        NOVA_SCOPED("bias.q", static_cast<std::int64_t>(kQKV * 4), bias_add(qb_, w.qb, kQKV, kQKV, q_));
        NOVA_SCOPED("bias.k", static_cast<std::int64_t>(kKV * 4), bias_add(kb_, w.kb, kKV,  kKV,  q_));
        NOVA_SCOPED("bias.v", static_cast<std::int64_t>(kKV * 4), bias_add(vb_, w.vb, kKV,  kKV,  q_));
    }

    // 4) RoPE + KV 落盘
    NOVA_SCOPED("rope", static_cast<std::int64_t>(kQKV * 4), rope_apply(qb_, kb_, pos0, cs_, sn_, q_));
    NOVA_SCOPED("kv_store", static_cast<std::int64_t>(kKV * 2 * 4), kv_store(kvk_ + kv_off, kvv_ + kv_off, kb_, vb_, 1, pos0, q_));

    // 5) attention（decode：n_past 与 CUDA 版同义，KV 里已有 pos0+1 个位置）
    //    bytes = 这一层 KV cache 里被读到的字节（随 pos 增长，带宽才有意义）
    NOVA_SCOPED("attn_decode", static_cast<std::int64_t>(pos0 + 1) * kKV * 2 * 4,
                attn_decode(qb_, kvk_ + kv_off, kvv_ + kv_off, ab_, pos0, scratch_, q_));

    // 6) quant(attn_out) + o 投影
    NOVA_SCOPED("quant_rows", static_cast<std::int64_t>(kD * 4), quant_rows<kD>(A_, ab_, 1, q_));
    NOVA_SCOPED("gemv.o", bo, gemv<kD, true>(w.o, A_, db_, q_));

    // 7) 残差
    NOVA_SCOPED("residual.attn", static_cast<std::int64_t>(kD * 2 * 4), residual_add(x_, db_, kD, q_));

    // 8) rms_norm(x, ffn_norm) + quant
    NOVA_SCOPED("rmsnorm_quant.ffn", static_cast<std::int64_t>(2 * kD * 4), rmsnorm_quant(h_, A_, x_, w.ffn_norm, 1, q_));

    // 9) gate / up
    NOVA_SCOPED("gemv.gate", bg, gemv<kD, true>(w.gate, A_, gb_, q_));
    NOVA_SCOPED("gemv.up",   bu, gemv<kD, true>(w.up,   A_, ub_, q_));

    // 10) swiglu(gate, up) -> gb 原地 + 量化成 FFN 宽激活 A2
    NOVA_SCOPED("swiglu", static_cast<std::int64_t>(kFFN * 4), swiglu_ffn(A2_, gb_, gb_, ub_, 1, q_));

    // 11) down
    NOVA_SCOPED("gemv.down", bd, gemv<kFFN, true>(w.down, A2_, db_, q_));

    // 12) 残差
    NOVA_SCOPED("residual.ffn", static_cast<std::int64_t>(kD * 2 * 4), residual_add(x_, db_, kD, q_));
}

void E2ERunner::head() {
    ::nova::scope::Scope::get().set_layer(-1);
    NOVA_SCOPED("rms_norm.head", static_cast<std::int64_t>(kD) * 4, rms_norm(h_, x_, out_norm_, 1, q_));
    NOVA_SCOPED("quant_rows.head", static_cast<std::int64_t>(kD) * 4, quant_rows<kD>(A_, h_, 1, q_));
    // 词表投影：这一发的权重字节就是整个输出矩阵（tie 时复用 embedding）
    if (tied_) {
        NOVA_SCOPED("gemv.head", static_cast<std::int64_t>(emb_.bytes()), gemv<kD, true>(emb_,  A_, logits_, q_));
    } else {
        NOVA_SCOPED("gemv.head", static_cast<std::int64_t>(outw_.bytes()), gemv<kD, true>(outw_, A_, logits_, q_));
    }
}

int E2ERunner::sample(bool greedy) {
    dev_.download(h_logits_.data(), logits_, static_cast<std::uint64_t>(kV) * 4);
    dev_.sync();

    if (greedy) {
        int best = 0;
        float bv = h_logits_[0];
        for (int i = 1; i < kV; ++i) {
            if (h_logits_[static_cast<std::size_t>(i)] > bv) { bv = h_logits_[static_cast<std::size_t>(i)]; best = i; }
        }
        return best;
    }

    // 与 CUDA 版 sample_last 同参：top-k 40 / top-p 0.95 / temp 0.8
    const int K = nova::kTopK < kV ? nova::kTopK : kV;
    std::vector<int> idx(static_cast<std::size_t>(kV));
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                     [&](int a, int b) { return h_logits_[static_cast<std::size_t>(a)] >
                                                h_logits_[static_cast<std::size_t>(b)]; });
    idx.resize(static_cast<std::size_t>(K));
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return h_logits_[static_cast<std::size_t>(a)] > h_logits_[static_cast<std::size_t>(b)];
    });

    const float mx = h_logits_[static_cast<std::size_t>(idx[0])];
    std::vector<float> pr(static_cast<std::size_t>(K));
    double sum = 0;
    for (int k = 0; k < K; ++k) {
        pr[static_cast<std::size_t>(k)] =
            std::exp((h_logits_[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])] - mx) / nova::kTemp);
        sum += pr[static_cast<std::size_t>(k)];
    }
    double acc = 0;
    int keep = K;
    for (int k = 0; k < K; ++k) {
        acc += pr[static_cast<std::size_t>(k)];
        if (acc >= nova::kTopP * sum) { keep = k + 1; break; }
    }

    rng_ = rng_ * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>((rng_ >> 11) & 0xFFFFFFFFull) / 4294967296.0 *
                     (nova::kTopP < 1.f ? acc : sum);
    double c = 0;
    int pick = idx[0];
    for (int k = 0; k < keep; ++k) {
        c += pr[static_cast<std::size_t>(k)];
        if (u <= c) { pick = idx[static_cast<std::size_t>(k)]; break; }
    }
    return pick;
}

int E2ERunner::step(int token_id, int pos, bool greedy) {
    check_ctx(pos);
    const auto t0 = Clock::now();

    embed(token_id);
    for (int l = 0; l < kL; ++l) layer(l, pos);
    head();
    const int next = sample(greedy);

    dev_.sync();                      // 计时到"这一次 decode 真的算完"为止
    last_ms_  = secs(t0, Clock::now()) * 1e3;
    last_tok_ = next;
    return next;
}

void E2ERunner::warmup(int n) {
    for (int i = 0; i < n; ++i) {
        step(9707, 0, true);          // 任意 token，只为让 kernel/驱动热起来
    }
}

} // namespace sycls
} // namespace nova
