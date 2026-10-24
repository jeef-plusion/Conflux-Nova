// nova_v2/code/device/sycl/dp4a_probe.cpp — 编译期探针（只编译、不运行）
//
// 目的：判断这个 oneAPI 版本的 sycl::dp4a 到底长什么样。build_sycl.bat 依次用
//   NOVA_PROBE=1 -> sycl::dp4a(vec<int8_t,4>, vec<int8_t,4>, int)
//   NOVA_PROBE=2 -> sycl::dp4a(vec<int8_t,4>, vec<int8_t,4>, vec<int32_t,1>)
// 试着编译这个文件，第一个编过的就 -DNOVA_SYCL_DP4A=<n> 传给正式构建；
// 两个都编不过就落回 0（纯标量）并大字告警——绝不静默降级。
#include <sycl/sycl.hpp>
#include <cstdint>

namespace {

sycl::vec<std::int8_t, 4> as8(std::int32_t w) {
    sycl::vec<std::int32_t, 1> v(w);
    return v.as<sycl::vec<std::int8_t, 4>>();
}

class Dp4aProbe {
public:
    void operator()(sycl::nd_item<1> it) const {
        const sycl::vec<std::int8_t, 4> a = as8(1);
        const sycl::vec<std::int8_t, 4> b = as8(2);
#if NOVA_PROBE == 1
        const int r = sycl::dp4a(a, b, 0);
#elif NOVA_PROBE == 2
        const sycl::vec<std::int32_t, 1> rv = sycl::dp4a(a, b, sycl::vec<std::int32_t, 1>(0));
        const int r = rv[0];
#else
        const int r = (a[0] * b[0]);
#endif
        if (it.get_global_id(0) == 0 && r == 12345678) {
            // 防止整段被优化掉；永远不会成立
            as8(r);
        }
    }
};

} // namespace

int main() {
    sycl::queue q;
    q.parallel_for<Dp4aProbe>(sycl::range<1>(32), Dp4aProbe{}).wait();
    return 0;
}
