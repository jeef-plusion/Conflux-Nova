// nova_v2/tests/host_sycl/sycl/sycl.hpp — 极简 SYCL 垫片（只给 CPU 上的 kernel 对拍用）
//
// 目的：在没有 Intel GPU / 没有 oneAPI 的机器上，把 src/kernel/sycl/*.cpp 里**真正的
//       kernel 体**跑起来，做数值对拍。它不是 SYCL 实现，也不代替目标机的编译与计时：
//       - 只实现 q4k_gemv.cpp / quant.cpp 用到的那一小撮 API；
//       - subgroup 归约用 double 顺序累加（GPU 上是树形归约），所以只会比真机更准；
//       - 线程 = work-item，barrier = 条件变量，local_accessor = 每个 WG 一块堆内存。
//
// 两条与真机一致的建模假设（改 kernel 时不要打破）：
//   1. work-group 内线性编号 = 最后一维最快（local = (R, SG) 时，一个 subgroup 就是
//      连续 SG 个 work-item，即固定 local_id(0)、local_id(1) 走 0..SG-1）。
//      现有 kernel（q4k_gemv / q4k_gemv4 / attn）都按这个假设写，并在目标机上对拍通过。
//   2. 一个 work-group 的 work-item 以「轮次」推进：每轮结束有一次 WG 屏障，
//      对应 GPU 上同一个 WG 不会有一部分 work-item 跑到下一轮的情况。
//
// 构建方式见 tests/sycl_gemv_host_check.cpp 文件头。
#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// MSVC 没有 __builtin_memcpy。kernel 里用它只是为了让 clang 折成一条 128-bit 载入。
#ifndef __builtin_memcpy
#  define __builtin_memcpy std::memcpy
#endif

// subgroup 大小必须与真机编译时的 NOVA_SYCL_SG 一致（kernel 用 reqd_sub_group_size 写死）
#ifndef NOVA_SYCL_SG
#  define NOVA_SYCL_SG 32
#endif
inline constexpr int kShimSgSize = NOVA_SYCL_SG;

namespace sycl {

// 数学函数：真 SYCL 在 sycl:: 里也有一套，这里直接借 std 的
using std::fabs;
using std::fmax;
using std::fmin;
using std::exp;
using std::fma;
using std::rint;
using std::sqrt;
inline float rsqrt(float x) { return 1.f / std::sqrt(x); }

// ===========================================================================
// range / nd_range / access
// ===========================================================================
template <int Dim>
class range {
public:
    range() {
        for (int i = 0; i < Dim; ++i) v_[i] = 1;
    }
    explicit range(std::size_t s) {
        static_assert(Dim == 1, "range<1> 只接受一个参数");
        v_[0] = s;
    }
    range(std::size_t a, std::size_t b) {
        static_assert(Dim == 2, "range<2> 只接受两个参数");
        v_[0] = a;
        v_[1] = b;
    }
    std::size_t operator[](int i) const { return v_[i]; }
    std::size_t get(int i) const { return v_[i]; }
    std::size_t size() const {
        std::size_t n = 1;
        for (int i = 0; i < Dim; ++i) n *= v_[i];
        return n;
    }

private:
    std::size_t v_[Dim] = {};
};

template <int Dim>
class nd_range {
public:
    nd_range(range<Dim> global, range<Dim> local) : g_(global), l_(local) {}
    range<Dim> get_global_range() const { return g_; }
    range<Dim> get_local_range() const { return l_; }

private:
    range<Dim> g_, l_;
};

namespace access {
enum class fence_space { local_space, global_space, global_and_local };
} // namespace access

// ===========================================================================
// 加性 / 最大值算子（reduce_over_group 用）。真 SYCL 里 plus<> / maximum<>
// 是 “template <class T = void>” 并对 void 特化；这里照抄这个形状。
// ===========================================================================
template <class T = void>
struct plus {
    T operator()(const T& a, const T& b) const { return a + b; }
};
template <>
struct plus<void> {
    template <class A, class B>
    auto operator()(const A& a, const B& b) const { return a + b; }
};

template <class T = void>
struct maximum {
    T operator()(const T& a, const T& b) const { return a < b ? b : a; }
};
template <>
struct maximum<void> {
    template <class A, class B>
    auto operator()(const A& a, const B& b) const { return a < b ? b : a; }
};

// ===========================================================================
// vec
// ===========================================================================
template <class T, int N>
class vec {
public:
    vec() = default;
    explicit vec(T s) {
        for (int i = 0; i < N; ++i) v_[i] = s;
    }
    T& operator[](int i) { return v_[i]; }
    T  operator[](int i) const { return v_[i]; }
    T& x() { return v_[0]; }
    T& y() { return v_[1]; }
    T& z() { return v_[2]; }
    T& w() { return v_[3]; }
    T  x() const { return v_[0]; }
    T  y() const { return v_[1]; }
    T  z() const { return v_[2]; }
    T  w() const { return v_[3]; }
    template <class U>
    U as() const {
        static_assert(sizeof(U) == sizeof(vec), "as<> 只做同宽重解释");
        U r;
        std::memcpy(&r, this, sizeof(U));
        return r;
    }

private:
    T v_[N] = {};
};

// ===========================================================================
// 屏障 + subgroup 上下文
// ===========================================================================
namespace detail {

class Barrier {
public:
    explicit Barrier(std::size_t n) : n_(n) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        const int gen = gen_;
        if (++count_ == static_cast<int>(n_)) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen_ != gen; });
        }
    }

private:
    std::mutex              m_;
    std::condition_variable cv_;
    std::size_t             n_;
    int                     count_ = 0;
    int                     gen_   = 0;
};

struct SgCtx {
    explicit SgCtx(std::size_t n) : size(n), barrier(n), vals(n, 0.0) {}
    std::size_t         size;
    Barrier             barrier;
    std::vector<double> vals;
};

} // namespace detail

class sub_group {
public:
    sub_group() = default;
    sub_group(detail::SgCtx* c, int lane) : ctx_(c), lane_(lane) {}
    int         get_local_linear_id() const { return lane_; }
    int         get_local_id() const { return lane_; }
    std::size_t get_local_range() const { return ctx_ != nullptr ? ctx_->size : 1; }
    // 垫片内部用；真 SYCL 没有这两个
    detail::SgCtx* detail_ctx() const { return ctx_; }
    int            detail_lane() const { return lane_; }

private:
    detail::SgCtx* ctx_  = nullptr;
    int            lane_ = 0;
};

template <class T>
inline T reduce_over_group(const sub_group& sg, T v, plus<>) {
    detail::SgCtx* c = sg.detail_ctx();
    c->vals[static_cast<std::size_t>(sg.detail_lane())] = static_cast<double>(v);
    c->barrier.wait();
    double s = 0.0;
    for (std::size_t i = 0; i < c->vals.size(); ++i) s += c->vals[i];
    c->barrier.wait();   // 读完再放行，避免下一轮覆盖
    return static_cast<T>(s);
}

template <class T>
inline T reduce_over_group(const sub_group& sg, T v, maximum<>) {
    detail::SgCtx* c = sg.detail_ctx();
    c->vals[static_cast<std::size_t>(sg.detail_lane())] = static_cast<double>(v);
    c->barrier.wait();
    double s = c->vals[0];
    for (std::size_t i = 1; i < c->vals.size(); ++i) s = (s < c->vals[i]) ? c->vals[i] : s;
    c->barrier.wait();
    return static_cast<T>(s);
}

template <class T>
inline T group_broadcast(const sub_group& /*sg*/, T v, std::size_t /*index*/) {
    // 垫片里所有 lane 拿的是同一个值，直接返回
    return v;
}

// ===========================================================================
// nd_item
// ===========================================================================
template <int Dim>
class nd_item {
public:
    std::size_t get_group(int /*d*/) const { return group_; }
    std::size_t get_local_id(int d) const { return d == 0 ? lid0_ : lid1_; }
    std::size_t get_local_linear_id() const { return lid0_ * lid1_range_ + lid1_; }
    std::size_t get_global_linear_id() const { return gid_linear_; }
    sub_group   get_sub_group() const { return sub_group(sg_, lane_in_sg_); }
    void        barrier(access::fence_space /*f*/) const { wg_barrier_->wait(); }

    // 垫片内部用
    void detail_set(std::size_t group, std::size_t lid0, std::size_t lid1, std::size_t lid1_range,
                    std::size_t gid_linear, detail::SgCtx* sg, int lane_in_sg, detail::Barrier* wg) {
        group_       = group;
        lid0_        = lid0;
        lid1_        = lid1;
        lid1_range_  = lid1_range;
        gid_linear_  = gid_linear;
        sg_          = sg;
        lane_in_sg_  = lane_in_sg;
        wg_barrier_  = wg;
    }

private:
    std::size_t      group_      = 0;
    std::size_t      lid0_       = 0;
    std::size_t      lid1_       = 0;
    std::size_t      lid1_range_ = 1;
    std::size_t      gid_linear_ = 0;
    detail::SgCtx*   sg_         = nullptr;
    int              lane_in_sg_ = 0;
    detail::Barrier* wg_barrier_ = nullptr;
};

// ===========================================================================
// 执行器：L 个线程 = 一个 work-group 的 L 个 work-item，逐 WG 推进
// ===========================================================================
namespace detail {

template <int Dim>
inline void local_coords(const range<Dim>& local, std::size_t t, std::size_t& lid0, std::size_t& lid1) {
    if constexpr (Dim == 1) {
        lid0 = t;
        lid1 = 0;
    } else {
        lid1 = t % local[1];   // 最后一维最快
        lid0 = t / local[1];
    }
}

template <int Dim>
inline std::size_t lid1_range_of(const range<Dim>& local) {
    if constexpr (Dim == 1) {
        return 1;
    } else {
        return local[1];
    }
}

template <int Dim, class F>
void execute(const nd_range<Dim>& nd, F f) {
    const std::size_t local_n    = nd.get_local_range().size();
    const std::size_t local_0    = nd.get_local_range()[0];
    const std::size_t groups     = nd.get_global_range()[0] / local_0;
    const std::size_t lid1_range = lid1_range_of<Dim>(nd.get_local_range());
    const std::size_t n_sg       = local_n / static_cast<std::size_t>(kShimSgSize);

    std::vector<std::unique_ptr<SgCtx>> sg_ctx;
    sg_ctx.reserve(n_sg);
    for (std::size_t i = 0; i < n_sg; ++i) {
        sg_ctx.emplace_back(new SgCtx(static_cast<std::size_t>(kShimSgSize)));
    }
    Barrier wg_barrier(local_n);

    std::vector<std::thread> pool;
    pool.reserve(local_n);
    for (std::size_t t = 0; t < local_n; ++t) {
        pool.emplace_back([&, t] {
            std::size_t lid0 = 0, lid1 = 0;
            local_coords<Dim>(nd.get_local_range(), t, lid0, lid1);
            SgCtx*       sg      = sg_ctx[t / static_cast<std::size_t>(kShimSgSize)].get();
            const int    lane_sg = static_cast<int>(t % static_cast<std::size_t>(kShimSgSize));
            nd_item<Dim> it;
            for (std::size_t g = 0; g < groups; ++g) {
                it.detail_set(g, lid0, lid1, lid1_range, g * local_0 + t, sg, lane_sg, &wg_barrier);
                f(it);
                wg_barrier.wait();   // 轮次之间保持 lockstep（GPU 上同一个 WG 不会跑飞）
            }
        });
    }
    for (auto& th : pool) th.join();
}

struct LocalBase {
    virtual ~LocalBase() {}
    virtual void bind()   = 0;
    virtual void unbind() = 0;
};

template <class T>
struct LocalImpl : LocalBase {
    explicit LocalImpl(std::size_t n) : n(n) {}
    void bind() override { p.reset(new T[n]); }
    void unbind() override { p.reset(); }
    std::size_t          n = 0;
    std::unique_ptr<T[]> p;
};

} // namespace detail

// ===========================================================================
// handler / local_accessor / queue / device / context
// ===========================================================================
class handler;

template <class T, int Dim>
class local_accessor {
public:
    local_accessor(range<Dim> r, handler& h);
    T&          operator[](std::size_t i) const { return impl_->p[i]; }
    std::size_t size() const { return impl_->n; }

private:
    std::shared_ptr<detail::LocalImpl<T>> impl_;
};

class handler {
public:
    template <class T>
    void add_local(const std::shared_ptr<detail::LocalImpl<T>>& l) {
        locals_.push_back(std::static_pointer_cast<detail::LocalBase>(l));
    }

    template <class KName, class F, int Dim>
    void parallel_for(const nd_range<Dim>& nd, F f) {
        for (auto& l : locals_) l->bind();
        detail::execute<Dim, F>(nd, f);
        for (auto& l : locals_) l->unbind();
    }

private:
    std::vector<std::shared_ptr<detail::LocalBase>> locals_;
};

template <class T, int Dim>
local_accessor<T, Dim>::local_accessor(range<Dim> r, handler& h)
    : impl_(std::make_shared<detail::LocalImpl<T>>(r.size())) {
    h.add_local(impl_);
}

class queue {
public:
    queue() = default;
    template <class F>
    void submit(F f) {
        handler h;
        f(h);
    }
    template <class KName, class F, int Dim>
    void parallel_for(const nd_range<Dim>& nd, F f) {
        handler h;
        h.template parallel_for<KName>(nd, f);
    }
    void wait() const {}
    void memset(void* p, int value, std::size_t bytes) { std::memset(p, value, bytes); }
    void memcpy(void* dst, const void* src, std::size_t bytes) { std::memcpy(dst, src, bytes); }
};

class device {
public:
    device() = default;
};

class context {
public:
    context() = default;
};

} // namespace sycl
