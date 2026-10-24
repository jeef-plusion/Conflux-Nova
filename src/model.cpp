// nova_v2/src/model.cpp — 权重上传 + 逐层 forward（工作区/流按会话传入）
#include "nova/model.h"
#include "nova/loader/validate.h"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <cmath>
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

size_t Engine::ws_bytes(int n) {
    return (size_t)n * ((size_t)Model::D * 4 * 5 + (size_t)Model::FFN * 4 * 2
                      + (size_t)Model::D + (size_t)Model::FFN
                      + (size_t)Model::D / 4 + (size_t)Model::FFN / 4)
         + (size_t)Model::V * 4 + (size_t)Model::D * 9 + (8u << 20);
}

void Engine::load(const std::string& path) {
    GgufFile g(path);
    verify_model(g);

    const size_t kvb = 2ull * kMaxSessions * (size_t)kv_span() * sizeof(float);
    const size_t total = Model::weight_bytes() + kvb + (size_t)kMaxSessions * ws_bytes(kSessCtx) + (128u << 20);
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

    kv_k_pool_ = arena_.alloc<float>((size_t)kMaxSessions * kv_span());
    kv_v_pool_ = arena_.alloc<float>((size_t)kMaxSessions * kv_span());
    bx  = arena_.alloc<float>(4ull * Model::D);
    bh  = arena_.alloc<float>(4ull * Model::D);
    bq  = arena_.alloc<float>(4ull * Model::NH * Model::HD);
    bk  = arena_.alloc<float>(4ull * Model::NKV * Model::HD);
    bv  = arena_.alloc<float>(4ull * Model::NKV * Model::HD);
    bab = arena_.alloc<float>(4ull * Model::NH * Model::HD);
    bg  = arena_.alloc<float>(4ull * Model::FFN);
    bu  = arena_.alloc<float>(4ull * Model::FFN);
    bd  = arena_.alloc<float>(4ull * Model::D);
    blogits = arena_.alloc<float>(4ull * Model::V);
    bact  = arena_.alloc<int8_t>(4ull * Model::D);
    bactd = arena_.alloc<float>(4ull * (Model::D / 32));
    basum = arena_.alloc<int>(4ull * (Model::D / 16));
    bact2 = arena_.alloc<int8_t>(4ull * Model::FFN);
    bact2d= arena_.alloc<float>(4ull * (Model::FFN / 32));
    basum2= arena_.alloc<int>(4ull * (Model::FFN / 16));
    bids  = arena_.alloc<int>(4);
    bpos  = arena_.alloc<int>(4);
    battn_scratch = arena_.alloc<float>((size_t)Model::NH * 32 * (Model::HD + 2));
    rcs_ = arena_.alloc<float>((size_t)kCtx * Model::HD / 2);
    rsn_ = arena_.alloc<float>((size_t)kCtx * Model::HD / 2);

    cudaFreeHost(pin);
    Kernel::rope_table(rcs_, rsn_, 0);
    cudaDeviceSynchronize();

    std::printf("[nova] 权重 %.3f GiB | 池 %.3f / %.3f GiB | KV %.3f GiB | ws/session %.1f MB\n",
                weight_bytes_ / 1073741824.0, arena_.used() / 1073741824.0, arena_.cap() / 1073741824.0,
                (double)kvb / 1073741824.0, ws_bytes(kSessCtx) / 1048576.0);
}

void Engine::make_workspace(Workspace* wp, int n_max, cudaStream_t s) {
    Workspace& w = *wp;
    w = Workspace{};
    w.s = s;
    w.n_max = n_max;
    w.x  = arena_.alloc<float>((size_t)n_max * Model::D);
    w.h  = arena_.alloc<float>((size_t)n_max * Model::D);
    w.qb = arena_.alloc<float>((size_t)n_max * Model::NH * Model::HD);
    w.kb = arena_.alloc<float>((size_t)n_max * Model::NKV * Model::HD);
    w.vb = arena_.alloc<float>((size_t)n_max * Model::NKV * Model::HD);
    w.ab = arena_.alloc<float>((size_t)n_max * Model::NH * Model::HD);
    w.gb = arena_.alloc<float>((size_t)n_max * Model::FFN);
    w.ub = arena_.alloc<float>((size_t)n_max * Model::FFN);
    w.db = arena_.alloc<float>((size_t)n_max * Model::D);
    w.act  = arena_.alloc<int8_t>((size_t)n_max * Model::D);
    w.actd = arena_.alloc<float>((size_t)n_max * (Model::D / 32));
    w.act2 = arena_.alloc<int8_t>((size_t)n_max * Model::FFN);
    w.act2d= arena_.alloc<float>((size_t)n_max * (Model::FFN / 32));
    w.asum = arena_.alloc<int>((size_t)n_max * (Model::D / 16));
    w.asum2= arena_.alloc<int>((size_t)n_max * (Model::FFN / 16));
    w.logits = arena_.alloc<float>((size_t)Model::V);
    w.attn_scratch = arena_.alloc<float>((size_t)Model::NH * 32 * (Model::HD + 2));
    w.ids = arena_.alloc<int>((size_t)n_max);
}

void Engine::prefill(Workspace& w, const int* ids, int n, float* kvk, float* kvv, int64_t pos0) {
    NOVA_REQUIRE(n <= w.n_max);
    cudaMemcpyAsync(w.ids, ids, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, w.s);
    Kernel::embed(w.x, tok_embd_, w.ids, n, w.s);
    for (int l = 0; l < Model::L; ++l) layer(w, l, n, pos0, kvk, kvv);
    head(w, n);
    cudaStreamSynchronize(w.s);
}

int Engine::decode(Workspace& w, int id, float* kvk, float* kvv, int64_t pos) {
    cudaMemcpyAsync(w.ids, &id, sizeof(int), cudaMemcpyHostToDevice, w.s);
    Kernel::embed(w.x, tok_embd_, w.ids, 1, w.s);
    for (int l = 0; l < Model::L; ++l) layer(w, l, 1, pos, kvk, kvv);
    head(w, 1);
    return sample_last(w);
}

int Engine::sample_last(Workspace& w) {
    if (h_logits_.empty()) h_logits_.resize((size_t)Model::V);
    cudaMemcpyAsync(h_logits_.data(), w.logits, (size_t)Model::V * sizeof(float),
                    cudaMemcpyDeviceToHost, w.s);
    cudaStreamSynchronize(w.s);

    const int V = (int)Model::V;
    const int K = kTopK < V ? kTopK : V;
    static std::vector<int> idx;
    idx.resize((size_t)V);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                     [&](int a, int b) { return h_logits_[a] > h_logits_[b]; });
    idx.resize((size_t)K);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return h_logits_[a] > h_logits_[b]; });

    const float mx = h_logits_[idx[0]];
    std::vector<float> pr((size_t)K);
    double sum = 0;
    for (int k = 0; k < K; ++k) {
        pr[k] = std::exp((h_logits_[idx[k]] - mx) / kTemp);
        sum += pr[k];
    }
    // top-p
    double acc = 0; int keep = K;
    for (int k = 0; k < K; ++k) { acc += pr[k]; if (acc >= kTopP * sum) { keep = k + 1; break; } }

    rng_ = rng_ * 6364136223846793005ull + 1442695040888963407ull;
    const double u = (double)((rng_ >> 11) & 0xFFFFFFFFull) / 4294967296.0 * (kTopP < 1.f ? acc : sum);
    double c = 0; int pick = idx[0];
    for (int k = 0; k < keep; ++k) { c += pr[k]; if (u <= c) { pick = idx[k]; break; } }

    idx.resize((size_t)V);
    std::iota(idx.begin(), idx.end(), 0);
    return pick;
}

void Engine::layer(Workspace& w, int l, int n, int64_t pos0, float* kvk, float* kvv) {
    const LayerW& lw = layers_[l];
    const ActQ8 A{w.act, w.actd, w.asum};
    const ActQ8 A2{w.act2, w.act2d, w.asum2};
    const int64_t kvoff = (int64_t)l * kMaxCtx * Model::NKV * Model::HD;

    prof.begin(PH_NORM);
    Kernel::rms_norm(w.h, w.x, lw.attn_norm, n, w.s);
    Kernel::quant(A, w.h, n, w.s);
    prof.end();

    prof.begin(PH_MM);
    Kernel::q_proj_n(lw.q, A, w.qb, n, w.s);
    Kernel::k_proj_n(lw.k, A, w.kb, n, w.s);
    Kernel::v_proj_n(lw.v, A, w.vb, n, w.s);
    prof.end();

    prof.begin(PH_ELEM);
    if (Model::qkv_bias) {
        Kernel::bias(w.qb, lw.qb, n * Model::NH * Model::HD, Model::NH * Model::HD, w.s);
        Kernel::bias(w.kb, lw.kb, n * Model::NKV * Model::HD, Model::NKV * Model::HD, w.s);
        Kernel::bias(w.vb, lw.vb, n * Model::NKV * Model::HD, Model::NKV * Model::HD, w.s);
    }
    prof.end();

    prof.begin(PH_ROPE);
    Kernel::rope_batch(w.qb, w.kb, n, pos0, rcs_, rsn_, w.s);
    Kernel::kv_store(kvk + kvoff, kvv + kvoff, w.kb, w.vb, n, pos0, w.s);
    prof.end();

    prof.begin(PH_ATTN);
    if (n == 1) Kernel::attn_decode(w.qb, kvk + kvoff, kvv + kvoff, w.ab, (int)pos0, w.attn_scratch, w.s);
    else        Kernel::attn_prefill(w.qb, kvk + kvoff, kvv + kvoff, w.ab, n, (int)pos0, w.s);
    prof.end();

    prof.begin(PH_MM);
    Kernel::quant(A, w.ab, n, w.s);
    Kernel::o_proj_n(lw.o, A, w.db, n, w.s);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::residual(w.x, w.db, n * Model::D, w.s);
    prof.end();

    prof.begin(PH_NORM);
    Kernel::rms_norm(w.h, w.x, lw.ffn_norm, n, w.s);
    Kernel::quant(A, w.h, n, w.s);
    prof.end();

    prof.begin(PH_MM);
    Kernel::gate_n(lw.gate, A, w.gb, n, w.s);
    Kernel::up_n(lw.up, A, w.ub, n, w.s);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::swiglu_ffn(A2, w.gb, w.gb, w.ub, n, w.s);
    prof.end();

    prof.begin(PH_MM);
    Kernel::down_n(lw.down, A2, w.db, n, w.s);
    prof.end();

    prof.begin(PH_ELEM);
    Kernel::residual(w.x, w.db, n * Model::D, w.s);
    prof.end();
}

void Engine::head(Workspace& w, int n) {
    const ActQ8 A{w.act, w.actd, w.asum};
    prof.begin(PH_HEAD);
    Kernel::rms_norm(w.h, w.x + (int64_t)(n - 1) * Model::D, out_norm_, 1, w.s);
    Kernel::quant(A, w.h, 1, w.s);
    Kernel::lm_head_n(out_w_, A, w.logits, 1, w.s);
    prof.end();
}

void Engine::decode_batch(BatchItem* it, int n, cudaStream_t s) {
    constexpr int N = 4;
    int toks[N];
    for (int i = 0; i < N; ++i) toks[i] = it[i].tok_in;
    cudaMemcpyAsync(bids, toks, sizeof(toks), cudaMemcpyHostToDevice, s);
    for (int i = 0; i < N; ++i) Kernel::embed(bx + i * Model::D, tok_embd_, bids + i, 1, s);

    auto A  = [&](int i) { return ActQ8{bact  + (size_t)i * Model::D,   bactd  + (size_t)i * (Model::D / 32),   basum  + (size_t)i * (Model::D / 16)}; };
    auto A2 = [&](int i) { return ActQ8{bact2 + (size_t)i * Model::FFN, bact2d + (size_t)i * (Model::FFN / 32), basum2 + (size_t)i * (Model::FFN / 16)}; };

    for (int l = 0; l < Model::L; ++l) {
        const LayerW& lw = layers_[l];
        const int64_t kvoff = (int64_t)l * kMaxCtx * Model::NKV * Model::HD;

        for (int i = 0; i < N; ++i) {
            Kernel::rms_norm(bh + i * Model::D, bx + i * Model::D, lw.attn_norm, 1, s);
            Kernel::quant(A(i), bh + i * Model::D, 1, s);
        }
        launch_gemv4<Model::D, Model::NH * Model::HD>(lw.q, bact, bactd, basum, bq, s);
        launch_gemv4<Model::D, Model::NKV * Model::HD>(lw.k, bact, bactd, basum, bk, s);
        launch_gemv4<Model::D, Model::NKV * Model::HD>(lw.v, bact, bactd, basum, bv, s);

        int pv[4] = {0, 0, 0, 0};
        for (int i = 0; i < N; ++i) {
            if (Model::qkv_bias) {
                Kernel::bias(bq + i * Model::NH * Model::HD, lw.qb, Model::NH * Model::HD, Model::NH * Model::HD, s);
                Kernel::bias(bk + i * Model::NKV * Model::HD, lw.kb, Model::NKV * Model::HD, Model::NKV * Model::HD, s);
                Kernel::bias(bv + i * Model::NKV * Model::HD, lw.vb, Model::NKV * Model::HD, Model::NKV * Model::HD, s);
            }
            Kernel::rope_batch(bq + i * Model::NH * Model::HD, bk + i * Model::NKV * Model::HD, 1, it[i].pos, rcs_, rsn_, s);
            Kernel::kv_store(it[i].kvk + kvoff, it[i].kvv + kvoff,
                             bk + i * Model::NKV * Model::HD, bv + i * Model::NKV * Model::HD, 1, it[i].pos, s);
            pv[i] = (int)it[i].pos;
        }
        for (int i = 0; i < N; ++i) {
            Kernel::attn_decode(bq + i * Model::NH * Model::HD, it[i].kvk + kvoff, it[i].kvv + kvoff,
                                bab + i * Model::NH * Model::HD, (int)it[i].pos, battn_scratch, s);
            Kernel::quant(A(i), bab + i * Model::NH * Model::HD, 1, s);
        }
        launch_gemv4<Model::NH * Model::HD, Model::D>(lw.o, bact, bactd, basum, bd, s);
        for (int i = 0; i < N; ++i) Kernel::residual(bx + i * Model::D, bd + i * Model::D, Model::D, s);

        for (int i = 0; i < N; ++i) {
            Kernel::rms_norm(bh + i * Model::D, bx + i * Model::D, lw.ffn_norm, 1, s);
            Kernel::quant(A(i), bh + i * Model::D, 1, s);
        }
        launch_gemv4<Model::D, Model::FFN>(lw.gate, bact, bactd, basum, bg, s);
        launch_gemv4<Model::D, Model::FFN>(lw.up, bact, bactd, basum, bu, s);
        for (int i = 0; i < N; ++i)
            Kernel::swiglu_ffn(A2(i), bg + i * Model::FFN, bg + i * Model::FFN, bu + i * Model::FFN, 1, s);
        launch_gemv4<Model::FFN, Model::D>(lw.down, bact2, bact2d, basum2, bd, s);
        for (int i = 0; i < N; ++i) Kernel::residual(bx + i * Model::D, bd + i * Model::D, Model::D, s);
    }

    for (int i = 0; i < N; ++i) {
        Kernel::rms_norm(bh + i * Model::D, bx + i * Model::D, out_norm_, 1, s);
        Kernel::quant(A(i), bh + i * Model::D, 1, s);
    }
    launch_gemv4<Model::D, Model::V>(out_w_, bact, bactd, basum, blogits, s);
    for (int i = 0; i < N; ++i) Kernel::sample(blogits + (size_t)i * Model::V, bids + i, s);
    cudaMemcpyAsync(toks, bids, sizeof(toks), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    for (int i = 0; i < N; ++i) it[i].tok_out = toks[i];
}

} // namespace nova
