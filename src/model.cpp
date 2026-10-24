// nova_v2/src/model.cpp — 权重上传 + 逐层 forward
#include "nova/model.h"
#include "nova/loader/validate.h"
#include <cstdio>
#include <fstream>
#include <vector>

namespace nova {
namespace {

void upload(std::ifstream& f, const GgufFile& g, const TensorInfo& t, void* dst, char* pin, size_t cap) {
    f.seekg((std::streamoff)(g.data_offset() + t.offset), std::ios::beg);
    if (!f) NOVA_ABORT("seek failed");
    uint64_t left = t.bytes, off = 0;
    while (left) {
        const size_t n = (size_t)(left > cap ? cap : left);
        f.read(pin, (std::streamsize)n);
        if (!f) NOVA_ABORT("short read while uploading weights");
        cudaMemcpy((char*)dst + off, pin, n, cudaMemcpyHostToDevice);
        off += n;
        left -= n;
    }
}

} // namespace

void Engine::load(const std::string& path, int max_tokens) {
    n_max_ = max_tokens;

    GgufFile g(path);
    verify_model(g);

    cudaStreamCreate(&s_);
    prof.init();

    // 池大小：权重 + KV + 工作区，一次性 cudaMalloc
    const size_t ws = (size_t)n_max_ * ((size_t)Model::D * 4 * 3 + (size_t)Model::FFN * 4 * 2 + (size_t)Model::D * 2
                    + (size_t)Model::D / 4 + (size_t)Model::FFN / 4)      // sums: 每 16 元素一个 int
                    + (size_t)Model::V * 4 + (size_t)Model::D * 9 + (64u << 20);
    const size_t total = Model::weight_bytes() + Model::kv_bytes() + ws + (128u << 20);
    arena_.init(total);

    std::ifstream f(path, std::ios::binary);
    if (!f) NOVA_ABORT("cannot reopen model");

    const size_t kPin = 32u << 20;
    char* pin = nullptr;
    cudaHostAlloc((void**)&pin, kPin, cudaHostAllocDefault);

    auto W = [&](const std::string& n) -> Q4KWeight {
        const TensorInfo* t = g.find(n);
        if (!t) NOVA_ABORT("missing tensor");
        void* p = arena_.alloc(t->bytes);
        upload(f, g, *t, p, pin, kPin);
        weight_bytes_ += t->bytes;
        return Q4KWeight{(const uint8_t*)p, (int64_t)t->ne[1], (int64_t)t->ne[0]};
    };
    auto F = [&](const std::string& n) -> const float* {
        const TensorInfo* t = g.find(n);
        if (!t) NOVA_ABORT("missing tensor");
        void* p = arena_.alloc(t->bytes);
        upload(f, g, *t, p, pin, kPin);
        return (const float*)p;
    };

    layers_.resize(Model::L);
    for (int i = 0; i < Model::L; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerW& w = layers_[i];
        w.attn_norm = F(p + "attn_norm.weight");
        w.ffn_norm  = F(p + "ffn_norm.weight");
        w.q = W(p + "attn_q.weight");
        w.k = W(p + "attn_k.weight");
        w.v = W(p + "attn_v.weight");
        w.o = W(p + "attn_output.weight");
        w.gate = W(p + "ffn_gate.weight");
        w.up   = W(p + "ffn_up.weight");
        w.down = W(p + "ffn_down.weight");
        if (Model::qkv_bias) {
            w.qb = F(p + "attn_q.bias");
            w.kb = F(p + "attn_k.bias");
            w.vb = F(p + "attn_v.bias");
        }
    }

    tok_embd_ = W("token_embd.weight");
    out_w_    = Model::tied ? tok_embd_ : W("output.weight");
    out_norm_ = F("output_norm.weight");

    kv_k_ = arena_.alloc<float>((size_t)Model::L * kCtx * Model::NKV * Model::HD);
    kv_v_ = arena_.alloc<float>((size_t)Model::L * kCtx * Model::NKV * Model::HD);
    attn_scratch_ = arena_.alloc<float>((size_t)Model::NH * 32 * (Model::HD + 2));

    x_  = arena_.alloc<float>((size_t)n_max_ * Model::D);
    h_  = arena_.alloc<float>((size_t)n_max_ * Model::D);
    qb_ = arena_.alloc<float>((size_t)n_max_ * Model::NH * Model::HD);
    kb_ = arena_.alloc<float>((size_t)n_max_ * Model::NKV * Model::HD);
    vb_ = arena_.alloc<float>((size_t)n_max_ * Model::NKV * Model::HD);
    ab_ = arena_.alloc<float>((size_t)n_max_ * Model::NH * Model::HD);
    gb_ = arena_.alloc<float>((size_t)n_max_ * Model::FFN);
    ub_ = arena_.alloc<float>((size_t)n_max_ * Model::FFN);
    db_ = arena_.alloc<float>((size_t)n_max_ * Model::D);
    act_  = arena_.alloc<int8_t>((size_t)n_max_ * Model::D);
    actd_ = arena_.alloc<float>((size_t)n_max_ * (Model::D / 32));
    act2_ = arena_.alloc<int8_t>((size_t)n_max_ * Model::FFN);
    act2d_= arena_.alloc<float>((size_t)n_max_ * (Model::FFN / 32));
    asum_  = arena_.alloc<int>((size_t)n_max_ * (Model::D / 16));
    asum2_ = arena_.alloc<int>((size_t)n_max_ * (Model::FFN / 16));
    logits_ = arena_.alloc<float>((size_t)Model::V);
    ids_    = arena_.alloc<int>((size_t)n_max_);
    rcs_ = arena_.alloc<float>((size_t)kCtx * Model::HD / 2);
    rsn_ = arena_.alloc<float>((size_t)kCtx * Model::HD / 2);

    cudaFreeHost(pin);
    Kernel::rope_table(const_cast<float*>(rcs_), const_cast<float*>(rsn_), s_);
    cudaStreamSynchronize(s_);

    std::printf("[nova] 权重上传完成: %.3f GiB | 池 %.3f / %.3f GiB | KV %.3f GiB\n",
                weight_bytes_ / 1073741824.0, arena_.used() / 1073741824.0, arena_.cap() / 1073741824.0,
                Model::kv_bytes() / 1073741824.0);
    cudaMemset(kv_k_, 0, (size_t)Model::L * kCtx * Model::NKV * Model::HD * 4);
    cudaMemset(kv_v_, 0, (size_t)Model::L * kCtx * Model::NKV * Model::HD * 4);
    n_past_ = 0;
}

void Engine::reset() {
    cudaMemsetAsync(kv_k_, 0, (size_t)Model::L * kCtx * Model::NKV * Model::HD * 4, s_);
    cudaMemsetAsync(kv_v_, 0, (size_t)Model::L * kCtx * Model::NKV * Model::HD * 4, s_);
    n_past_ = 0;
}

void Engine::prefill(const int* ids, int n) {
    cudaMemcpyAsync(ids_, ids, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, s_);
    prof.begin(PH_EMBED);
    Kernel::embed(x_, tok_embd_, ids_, n, s_);
    prof.end();

    for (int l = 0; l < Model::L; ++l) layer(l, n, 0);
    head(n);
    cudaStreamSynchronize(s_);
    n_past_ = n;
}

int Engine::decode(int id) {
    cudaMemcpyAsync(ids_, &id, sizeof(int), cudaMemcpyHostToDevice, s_);
    prof.begin(PH_EMBED);
    Kernel::embed(x_, tok_embd_, ids_, 1, s_);
    prof.end();

    for (int l = 0; l < Model::L; ++l) layer(l, 1, n_past_);
    head(1);
    Kernel::sample(logits_, ids_, s_);
    int tok = 0;
    cudaMemcpyAsync(&tok, ids_, sizeof(int), cudaMemcpyDeviceToHost, s_);
    cudaStreamSynchronize(s_);
    ++n_past_;
    return tok;
}

void Engine::layer(int l, int n, int64_t pos0) {
    const LayerW& w = layers_[l];
    const ActQ8 A{act_, actd_, asum_};
    const ActQ8 A2{act2_, act2d_, asum2_};
    const int64_t kvoff = (int64_t)l * kCtx * Model::NKV * Model::HD;

    prof.begin(PH_NORM);
    Kernel::rms_norm(h_, x_, w.attn_norm, n, s_);
    Kernel::quant(ActQ8{act_, actd_, asum_}, h_, n, s_);
    prof.end();

    prof.begin(PH_MM);
    Kernel::q_proj_n(w.q, A, qb_, n, s_);
    Kernel::k_proj_n(w.k, A, kb_, n, s_);
    Kernel::v_proj_n(w.v, A, vb_, n, s_);
    prof.end();

    prof.begin(PH_ELEM);
    if (Model::qkv_bias) {
        Kernel::bias(qb_, w.qb, n * Model::NH * Model::HD, Model::NH * Model::HD, s_);
        Kernel::bias(kb_, w.kb, n * Model::NKV * Model::HD, Model::NKV * Model::HD, s_);
        Kernel::bias(vb_, w.vb, n * Model::NKV * Model::HD, Model::NKV * Model::HD, s_);
    }
    prof.end();

    prof.begin(PH_ROPE);
    Kernel::rope_batch(qb_, kb_, n, pos0, rcs_, rsn_, s_);
    Kernel::kv_store(kv_k_ + kvoff, kv_v_ + kvoff, kb_, vb_, n, pos0, s_);
    prof.end();

    prof.begin(PH_ATTN);
    if (n == 1) Kernel::attn_decode(qb_, kv_k_ + kvoff, kv_v_ + kvoff, ab_, (int)pos0, attn_scratch_, s_);
    else        Kernel::attn_prefill(qb_, kv_k_ + kvoff, kv_v_ + kvoff, ab_, n, (int)pos0, s_);
    prof.end();

    prof.begin(PH_MM);
    Kernel::quant(ActQ8{act_, actd_, asum_}, ab_, n, s_);
    Kernel::o_proj_n(w.o, A, db_, n, s_);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::residual(x_, db_, n * Model::D, s_);
    prof.end();

    prof.begin(PH_NORM);
    Kernel::rms_norm(h_, x_, w.ffn_norm, n, s_);
    Kernel::quant(ActQ8{act_, actd_, asum_}, h_, n, s_);
    prof.end();

    prof.begin(PH_MM);
    Kernel::gate_n(w.gate, A, gb_, n, s_);
    Kernel::up_n(w.up, A, ub_, n, s_);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::swiglu_ffn(A2, gb_, gb_, ub_, n, s_);
    prof.end();

    prof.begin(PH_MM);
    Kernel::down_n(w.down, A2, db_, n, s_);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::residual(x_, db_, n * Model::D, s_);
    prof.end();
}

void Engine::head(int n) {
    const ActQ8 A{act_, actd_, asum_};
    prof.begin(PH_HEAD);
    Kernel::rms_norm(h_, x_ + (int64_t)(n - 1) * Model::D, out_norm_, 1, s_);
    Kernel::quant(ActQ8{act_, actd_, asum_}, h_, 1, s_);
    Kernel::lm_head_n(out_w_, A, logits_, 1, s_);
    prof.end();
}

} // namespace nova
