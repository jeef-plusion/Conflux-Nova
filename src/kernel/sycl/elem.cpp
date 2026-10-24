// nova_v2/src/kernel/sycl/elem.cpp — embed / residual / swiglu / bias / kv_store
//
// 对照：
//   embed      <- src/kernel/elem.cu : embed_kernel（Q4_K 取一行并反量化）
//   kv_store   <- src/kernel/elem.cu : kv_store_kernel
//   bias       <- src/kernel/elem.cu : bias_kernel（改成 2D，消掉 i % cols）
//   residual   <- src/kernel/norm.cu : add_kernel
//   swiglu     <- src/kernel/norm.cu : swiglu_kernel + quant_kernel
//                 **这里融合成一个 kernel**：decode 每层省一次 dispatch，且 y 只写一次。
#include "kernels.h"

namespace nova {
namespace syclk {
namespace {

inline constexpr int kSgPerWg = kRmsWg / kSyclSg;

// --------------------------------------------------------------------------
// embed：一个 work-group 吃一个 token，256 个 work-item 正好覆盖一个 super-block
// --------------------------------------------------------------------------
class EmbedKernel {
public:
    EmbedKernel(float* x, const std::uint8_t* emb, const int* ids, int n)
        : x_(x), emb_(emb), ids_(ids), n_(n) {}

    void operator()(sycl::nd_item<2> it) const {
        constexpr int BPR = kD / kQK;
        const int t = static_cast<int>(it.get_group(0));
        const int i = static_cast<int>(it.get_local_id(1));   // local = (1, 256)：dim1 才是元素号
        if (t >= n_) return;

        const block_q4_K* row =
            reinterpret_cast<const block_q4_K*>(emb_) + static_cast<std::int64_t>(ids_[t]) * BPR;
        const int g   = i >> 6;
        const int r   = i & 63;
        const int sub = r >> 5;
        const int e   = r & 31;

        float* dst = x_ + static_cast<std::int64_t>(t) * kD;
        #pragma unroll 1
        for (int sb = 0; sb < BPR; ++sb) {
            const block_q4_K* b = row + sb;
            std::uint8_t sc, m;
            get_scale_min_k4(g * 2 + sub, b->scales, sc, m);
            const int byte = b->qs[g * 32 + e];
            const int q    = sub ? (byte >> 4) : (byte & 0xF);
            dst[sb * kQK + i] = h2f(b->d) * static_cast<float>(sc) * static_cast<float>(q) -
                                h2f(b->dmin) * static_cast<float>(m);
        }
    }

private:
    float*              x_;
    const std::uint8_t* emb_;
    const int*          ids_;
    int                 n_;
};

class ResidualKernel {
public:
    ResidualKernel(float* acc, const float* x, int n) : acc_(acc), x_(x), n_(n) {}
    void operator()(sycl::nd_item<1> it) const {
        const int i = static_cast<int>(it.get_global_linear_id());
        if (i < n_) acc_[i] += x_[i];
    }
private:
    float*       acc_;
    const float* x_;
    int          n_;
};

class BiasKernel {
public:
    BiasKernel(float* y, const float* b, int cols) : y_(y), b_(b), cols_(cols) {}
    void operator()(sycl::nd_item<2> it) const {
        const std::int64_t row = static_cast<std::int64_t>(it.get_group(0));
        float* yr = y_ + row * cols_;
        for (int c = static_cast<int>(it.get_local_id(1)); c < cols_; c += kRmsWg) yr[c] += b_[c];
    }
private:
    float*       y_;
    const float* b_;
    int          cols_;
};

class KvStoreKernel {
public:
    KvStoreKernel(float* kc, float* vc, const float* k, const float* v, std::int64_t pos0)
        : kc_(kc), vc_(vc), k_(k), v_(v), pos0_(pos0) {}
    void operator()(sycl::nd_item<2> it) const {
        const std::int64_t gid = static_cast<std::int64_t>(it.get_group(0));   // t * NKV + h
        const int t   = static_cast<int>(gid / kNKV);
        const int h   = static_cast<int>(gid - static_cast<std::int64_t>(t) * kNKV);
        const int d   = static_cast<int>(it.get_local_id(1));
        const std::int64_t pos = pos0_ + t;
        const std::int64_t off = (pos * kNKV + h) * kHD + d;
        const std::int64_t src = (static_cast<std::int64_t>(t) * kNKV + h) * kHD + d;
        kc_[off] = k_[src];
        vc_[off] = v_[src];
    }
private:
    float*       kc_;
    float*       vc_;
    const float* k_;
    const float* v_;
    std::int64_t pos0_;
};

// --------------------------------------------------------------------------
// swiglu + quant 融合：一个 work-group 吃一行，subgroup 负责 32 元素块
// --------------------------------------------------------------------------
template <int K>
class SwiGLUQuantKernel {
public:
    SwiGLUQuantKernel(std::int8_t* qs, float* d, std::int32_t* s, float* y,
                      const float* gate, const float* up, int rows)
        : qs_(qs), d_(d), s_(s), y_(y), gate_(gate), up_(up), rows_(rows) {}

    void operator()(sycl::nd_item<1> it) const [[sycl::reqd_sub_group_size(kSyclSg)]] {
        constexpr int NB = K / 32;
        const int row = static_cast<int>(it.get_group(0));
        if (row >= rows_) return;
        const int tid = static_cast<int>(it.get_local_id(0));
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgid = tid / kSyclSg;

        const std::int64_t base = static_cast<std::int64_t>(row) * K;
        for (int b = sgid; b < NB; b += kSgPerWg) {
            const int   i = b * 32 + lane;
            const float g = gate_[base + i];
            const float u = up_[base + i];
            const float v = (g / (1.f + sycl::exp(-g))) * u;
            y_[base + i] = v;

            const float amax = sg_max<float>(sg, sycl::fabs(v));
            const float sc   = amax * (1.f / 127.f);
            const float inv  = (amax > 0.f) ? (127.f / amax) : 0.f;
            const int   qi   = static_cast<int>(sycl::rint(v * inv));
            qs_[base + i] = static_cast<std::int8_t>(qi);

            const int s_lo = sg_sum<int>(sg, (lane < 16) ? qi : 0);
            const int s_hi = sg_sum<int>(sg, (lane >= 16) ? qi : 0);
            if (lane == 0) {
                d_[base / 32 + b] = sc;
                const std::int64_t sb = (base / 32 + b) * 2;
                s_[sb + 0] = s_lo;
                s_[sb + 1] = s_hi;
            }
        }
    }

private:
    std::int8_t*  qs_;
    float*        d_;
    std::int32_t* s_;
    float*        y_;
    const float*  gate_;
    const float*  up_;
    int           rows_;
};

template <int K>
void swiglu_launch(ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q) {
    if (rows <= 0) return;
    q.parallel_for<SwiGLUQuantKernel<K>>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(rows) * kRmsWg), sycl::range<1>(kRmsWg)),
        SwiGLUQuantKernel<K>(dst.qs, dst.d, dst.s, y, gate, up, rows));
}

} // namespace

void embed_rows(float* x, const std::uint8_t* emb_q4k, const int* ids, int n, sycl::queue& q) {
    if (n <= 0) return;
    q.parallel_for<EmbedKernel>(
        sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(n), kQK), sycl::range<2>(1, kQK)),
        EmbedKernel(x, emb_q4k, ids, n));
}

void residual_add(float* acc, const float* x, int n, sycl::queue& q) {
    if (n <= 0) return;
    const int groups = div_up(n, kRmsWg);
    q.parallel_for<ResidualKernel>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(groups) * kRmsWg), sycl::range<1>(kRmsWg)),
        ResidualKernel(acc, x, n));
}

void bias_add(float* y, const float* b, int n, int cols, sycl::queue& q) {
    if (n <= 0 || cols <= 0) return;
    const std::size_t rows = static_cast<std::size_t>(n / cols);
    q.parallel_for<BiasKernel>(sycl::nd_range<2>(sycl::range<2>(rows, kRmsWg), sycl::range<2>(1, kRmsWg)),
                               BiasKernel(y, b, cols));
}

void kv_store(float* kc, float* vc, const float* k, const float* v, int n, std::int64_t pos0, sycl::queue& q) {
    if (n <= 0) return;
    q.parallel_for<KvStoreKernel>(
        sycl::nd_range<2>(sycl::range<2>(static_cast<std::size_t>(n) * kNKV, kHD), sycl::range<2>(1, kHD)),
        KvStoreKernel(kc, vc, k, v, pos0));
}

void swiglu(ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q) {
    swiglu_launch<kD>(dst, y, gate, up, rows, q);
}

void swiglu_ffn(ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q) {
    swiglu_launch<kFFN>(dst, y, gate, up, rows, q);
}

template <int K>
void swiglu_quant(ActQ8View dst, float* y, const float* gate, const float* up, int rows, sycl::queue& q) {
    swiglu_launch<K>(dst, y, gate, up, rows, q);
}

template void swiglu_quant<kD  >(ActQ8View, float*, const float*, const float*, int, sycl::queue&);
template void swiglu_quant<kFFN>(ActQ8View, float*, const float*, const float*, int, sycl::queue&);

} // namespace syclk
} // namespace nova
