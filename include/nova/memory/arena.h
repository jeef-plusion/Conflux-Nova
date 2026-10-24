// nova_v2/memory/arena.h — 唯一一次 cudaMalloc，之后只有 bump；溢出即断
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "nova/config.h"

namespace nova {

// 任何一个 CUDA 返回码都必须当场炸掉。
// 为什么：上下文一旦被污染（分配失败 / 非法地址），之后的 kernel 全部空转、
// 采样读到的是过期 logits，进程不会报错，只会以 1000+ tok/s 打印垃圾 token。
// 静默的坏结果比中止危险得多，所以这里把返回码变成唯一允许的失败方式。
inline void cu_ok(cudaError_t e, const char* what, const char* file, int line) {
    if (e == cudaSuccess) return;
    std::fflush(stdout);
    std::fprintf(stderr, "[nova] FATAL %s:%d  %s -> %s\n", file, line, what, cudaGetErrorString(e));
    if (e == cudaErrorMemoryAllocation) {
        std::size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess)
            std::fprintf(stderr, "[nova]       设备内存不足：free %.3f / total %.3f GiB\n",
                         fb / 1073741824.0, tb / 1073741824.0);
    }
    std::fflush(stderr);
    std::abort();
}
#define NOVA_CU(call) ::nova::cu_ok((call), #call, __FILE__, __LINE__)

class DeviceArena {
public:
    // 初始化时调用一次。整机的权重、KV、scratch 全从这里切。
    void init(std::size_t bytes) {
        NOVA_REQUIRE(base_ == nullptr);
        NOVA_REQUIRE(bytes <= (std::size_t)7ull * 1024 * 1024 * 1024 + kArenaReserve);
        base_ = nullptr;
        // 分配失败时 cudaMalloc 会把 base_ 置空；不检查就会得到 base_+offset 这种
        // 假指针，之后所有 kernel/拷贝全部失败而无人知晓（OOM 退化成垃圾输出）。
        NOVA_CU(cudaMalloc(&base_, bytes));
        NOVA_REQUIRE(base_ != nullptr);
        cap_ = bytes;
        off_ = 0;
    }

    // 偏移在 load 期一次算定，热路径只做指针加法
    void* alloc(std::size_t bytes, std::size_t align = 256) {
        std::size_t o = (off_ + align - 1) & ~(align - 1);
        NOVA_REQUIRE(o + bytes <= cap_);      // 池不够 = 维度或预算错了，直接炸
        void* p = base_ + o;
        off_ = o + bytes;
        return p;
    }

    template <class T>
    T* alloc(std::size_t count) { return static_cast<T*>(alloc(count * sizeof(T), 256)); }

    std::size_t used() const { return off_; }
    std::size_t cap()  const { return cap_; }
    char* base() const { return base_; }

private:
    char*       base_ = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
};

} // namespace nova
