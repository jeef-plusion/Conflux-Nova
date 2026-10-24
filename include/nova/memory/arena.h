// nova_v2/memory/arena.h — 唯一一次 cudaMalloc，之后只有 bump；溢出即断
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "nova/config.h"

namespace nova {

class DeviceArena {
public:
    // 初始化时调用一次。整机的权重、KV、scratch 全从这里切。
    void init(std::size_t bytes) {
        NOVA_REQUIRE(base_ == nullptr);
        NOVA_REQUIRE(bytes <= (std::size_t)7ull * 1024 * 1024 * 1024 + kArenaReserve);
        cudaMalloc(&base_, bytes);
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
