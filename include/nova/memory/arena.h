// nova_v2/memory/arena.h — 唯一的显存来源：预算是 load 期算好的，分配按块做，之后只有 bump
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// ---------------------------------------------------------------------------
// 为什么按块分配（2026-11-10 在 1080 Ti 上的实测）：
//   同一张卡、空闲 9.08 GiB，单次 cudaMalloc 的表现是
//       1.00 GiB -> ok
//       1.66 GiB -> out of memory
//   也就是 WDDM 下存在"单次分配上限"（远小于空闲量）。原来 init() 一次要 1.66 GiB
//   （权重 828 + KV 448 + 工作区 260 + 预留 128），直接 OOM，模型根本加载不了。
//   所以改成：预算照旧一次算好（cap() 仍是那个数），实际分配切成 512 MiB 的块，
//   每次 alloc 先看当前块放不放得下，放不下就开新块。语义与"一整块"完全一致
//   （所有指针都来自 alloc()，热路径仍是指针加法），只是绕开了单次上限。
// ---------------------------------------------------------------------------
class DeviceArena {
public:
    static constexpr std::size_t kChunkBytes = 512ull * 1024 * 1024;

    // 只记预算，不分配（分配推迟到第一次 alloc，便于把 OOM 精确归因到某个张量）
    void init(std::size_t bytes) {
        NOVA_REQUIRE(chunks_.empty());
        NOVA_REQUIRE(bytes <= (std::size_t)7ull * 1024 * 1024 * 1024 + kArenaReserve);
        budget_ = bytes;
        total_alloc_ = 0;
        total_used_ = 0;
    }

    void* alloc(std::size_t bytes, std::size_t align = 256) {
        NOVA_REQUIRE(align <= 256);            // 新块天然 256 B 对齐，块内按 align 上调
        if (!chunks_.empty()) {
            Chunk& c = chunks_.back();
            const std::size_t o = (c.used + align - 1) & ~(align - 1);
            if (o + bytes <= c.size) {
                c.used = o + bytes;
                total_used_ += bytes;
                return c.base + o;
            }
        }
        std::size_t sz = bytes + align;
        if (sz < kChunkBytes) sz = kChunkBytes;
        NOVA_REQUIRE(total_alloc_ + sz <= budget_ + kChunkBytes);
        if (std::getenv("NOVA_ARENA_DEBUG")) {
            std::printf("[arena] +chunk %.1f MiB (累计已分配 %.1f / 预算 %.1f MiB)\n",
                        (double)sz / 1048576.0, (double)(total_alloc_ + sz) / 1048576.0,
                        (double)budget_ / 1048576.0);
        }
        Chunk c{};
        c.size = sz;
        NOVA_CU(cudaMalloc(&c.base, sz));
        NOVA_REQUIRE(c.base != nullptr);
        c.used = bytes;                        // 从块首开始，天然对齐
        chunks_.push_back(c);
        total_alloc_ += sz;
        total_used_ += bytes;
        return c.base;
    }

    template <class T>
    T* alloc(std::size_t count) { return static_cast<T*>(alloc(count * sizeof(T), 256)); }

    std::size_t used() const { return total_used_; }      // 真实占用（各块用量之和）
    std::size_t cap()  const { return budget_; }          // 预算
    std::size_t reserved() const { return total_alloc_; } // 实际向驱动要到的量
    char* base() const { return chunks_.empty() ? nullptr : chunks_.front().base; }

private:
    struct Chunk {
        char*       base = nullptr;
        std::size_t size = 0;
        std::size_t used = 0;
    };
    std::vector<Chunk> chunks_;
    std::size_t budget_      = 0;
    std::size_t total_alloc_ = 0;
    std::size_t total_used_  = 0;
};

} // namespace nova
