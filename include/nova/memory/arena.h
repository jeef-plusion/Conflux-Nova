// nova_v2/memory/arena.h — 唯一的显存来源：分块申请 + 三级兜底 + 触碰预热
//
// 2026-11-10 在 1080 Ti / WDDM 上实测出来的三条硬事实，决定了这里的长相：
//
//  1. 单次大申请会被"动态预算"卡住：空闲 9 GiB 时单次 4 GiB 能成，但**同时持有**到
//     约 2 GiB 后第 9 个 256 MiB 块就 OOM（见 build\lt\oom_sweep.cu）。所以：
//     把预算切成 ≤512 MiB 的块，逐块申请；单块失败只影响一块。
//  2. WDDM 下桌面合成器会周期性刷新、被抢走的额度过几百毫秒往往归还 → 失败后
//     重试 kRetries 次、每次等 kRetryMs（默认 5 × 100 ms）再放弃。
//  3. **托管内存救不了 Pascal**：本机 concurrentManagedAccess = 0
//     （pageableMemoryAccess 也是 0），cudaMallocManaged 仍然要求装得下，
//     超额订阅是 Turing+ / Linux HMM 的能力。所以第三级兜底是
//     **cudaHostAllocMapped（映射主机内存）**：UVA 下内核可直接解引用，权重走 PCIe，
//     慢一个数量级，但"多大的模型都能塞进去"。
//
// 防弹衣：任何 CUDA 返回码都当场炸（NOVA_CU）；每块到手立刻 memset 触碰全部页，
// 把 page fault 从"推理中途"挪到"加载期"（托管/映射内存尤其明显）。
// 绝不允许 cudaMalloc 返回 NULL 之后继续用 —— 那正是 TDR / 0x12 的来源。
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "nova/config.h"

namespace nova {

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
    static constexpr std::size_t kChunkBytes = 512ull * 1024 * 1024;   // 512 MiB 一块
    static constexpr int         kRetries    = 5;
    static constexpr int         kRetryMs    = 100;

    explicit DeviceArena(bool verbose = true) : verbose_(verbose) {}

    // 禁止拷贝/移动：arena 拥有 chunk 所有权，拷贝一次就会 double free（代码审查 P0）。
    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&&)                 = delete;
    DeviceArena& operator=(DeviceArena&&)      = delete;

    ~DeviceArena() {
        for (auto& c : chunks_) {
            if (!c.base) continue;
            if (c.kind == Kind::HostMapped) cudaFreeHost(c.base);
            else                            cudaFree(c.base);
        }
        chunks_.clear();
    }

    void init(std::size_t bytes) {
        NOVA_REQUIRE(chunks_.empty());
        NOVA_REQUIRE(bytes <= (std::size_t)7ull * 1024 * 1024 * 1024 + kArenaReserve);
        budget_ = bytes;
        total_alloc_ = 0;
        total_used_ = 0;

        // llama.cpp 式：先一次性要一大块。2026-11-10 实测本机"单次上限 == 累计上限"
        // （arena_ceiling.exe：两者都是 2.25 GiB），说明不存在碎片问题，那就没必要
        // 一开始就切成小块 —— 一块到手后，内部的 bump 分配比多次 cudaMalloc 更省驱动开销。
        Chunk one{};
        one.size = bytes;
        one.kind = Kind::Device;
        void* big = nullptr;
        if (cudaMalloc(&big, bytes) == cudaSuccess && big != nullptr) {
            one.base = static_cast<char*>(big);
            one.used = 0;
            chunks_.push_back(one);
            total_alloc_ = bytes;
            if (verbose_) {
                std::printf("[arena] 一次性拿到 %.1f MiB 连续块（llama.cpp 式预分配）\n",
                            (double)bytes / 1048576.0);
            }
            return;
        }
        cudaGetLastError();
        if (verbose_) {
            std::printf("[arena] 一次性 %.1f MiB 失败，退回分块（每块 %.0f MiB）\n",
                        (double)bytes / 1048576.0, (double)kChunkBytes / 1048576.0);
        }
    }

    void* alloc(std::size_t bytes, std::size_t align = 256) {
        NOVA_REQUIRE(align <= 256);
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

        Chunk c{};
        c.size = sz;
        c.base = take_block(sz, c.kind);
        NOVA_REQUIRE(c.base != nullptr);
        c.used = bytes;
        chunks_.push_back(c);
        total_alloc_ += sz;
        total_used_ += bytes;
        return c.base;
    }

    template <class T>
    T* alloc(std::size_t count) { return static_cast<T*>(alloc(count * sizeof(T), 256)); }

    std::size_t used() const { return total_used_; }
    std::size_t cap()  const { return budget_; }
    std::size_t reserved() const { return total_alloc_; }
    int n_malloc() const { int n = 0; for (auto& c : chunks_) if (c.kind == Kind::Device) ++n; return n; }
    int n_managed() const { int n = 0; for (auto& c : chunks_) if (c.kind == Kind::Managed) ++n; return n; }
    int n_host() const { int n = 0; for (auto& c : chunks_) if (c.kind == Kind::HostMapped) ++n; return n; }
    char* base() const { return chunks_.empty() ? nullptr : chunks_.front().base; }

private:
    enum class Kind { Device, Managed, HostMapped };

    struct Chunk {
        char*       base = nullptr;
        std::size_t size = 0;
        std::size_t used = 0;
        Kind        kind = Kind::Device;
    };

    // 三 级 兜 底：cudaMalloc（重试）→ cudaMallocManaged → cudaHostAllocMapped
    char* take_block(std::size_t sz, Kind& kind_out) {
        kind_out = Kind::Device;
        cudaError_t last = cudaSuccess;

        for (int i = 0; i < kRetries; ++i) {
            void* p = nullptr;
            last = cudaMalloc(&p, sz);
            if (last == cudaSuccess && p != nullptr) {
                warm_device(p, sz);
                return static_cast<char*>(p);
            }
            cudaGetLastError();
            if (last != cudaErrorMemoryAllocation) break;
            if (verbose_) {
                std::printf("[arena] cudaMalloc %.1f MiB 失败（%d/%d），等 %d ms 让桌面把额度还回来\n",
                            (double)sz / 1048576.0, i + 1, kRetries, kRetryMs);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
        }

        void* m = nullptr;
        const cudaError_t em = cudaMallocManaged(&m, sz, cudaMemAttachGlobal);
        if (em == cudaSuccess && m != nullptr) {
            kind_out = Kind::Managed;
            if (verbose_) {
                std::printf("[arena] 退到托管内存 %.1f MiB%s\n", (double)sz / 1048576.0,
                            managed_oversubscribes() ? "" : "（本卡 concurrentManagedAccess=0，不能超额订阅，只是换个分配器）");
            }
            warm_device(m, sz);
            return static_cast<char*>(m);
        }
        cudaGetLastError();

        void* h = nullptr;
        const cudaError_t eh = cudaHostAlloc(&h, sz, cudaHostAllocMapped);
        if (eh == cudaSuccess && h != nullptr) {
            kind_out = Kind::HostMapped;
            std::memset(h, 0, sz);                 // 主机侧预置，避免内核首次读缺页
            if (verbose_) {
                std::printf("[arena] 退到映射主机内存 %.1f MiB —— 内核可直读，但走 PCIe（约 10 GB/s）\n",
                            (double)sz / 1048576.0);
            }
            return static_cast<char*>(h);
        }
        cudaGetLastError();

        std::fflush(stdout);
        std::fprintf(stderr,
                     "[arena] FATAL 无法分配 %.1f MiB：cudaMalloc=%s，cudaMallocManaged=%s，cudaHostAlloc=%s\n",
                     (double)sz / 1048576.0, last == cudaSuccess ? "?" : cudaGetErrorString(last),
                     cudaGetErrorString(em), cudaGetErrorString(eh));
        return nullptr;
    }

    static bool managed_oversubscribes() {
        static const bool ok = [] {
            int dev = 0, v = 0;
            if (cudaGetDevice(&dev) != cudaSuccess) return false;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, dev) != cudaSuccess) return false;
            return v != 0;
        }();
        return ok;
    }

    static void warm_device(void* p, std::size_t sz) {
        cudaMemset(p, 0, sz);                      // 设备侧写一遍：page fault 前移
        cudaGetLastError();
    }

    std::vector<Chunk> chunks_;
    std::size_t budget_      = 0;
    std::size_t total_alloc_ = 0;
    std::size_t total_used_  = 0;
    bool        verbose_     = true;
};

} // namespace nova
