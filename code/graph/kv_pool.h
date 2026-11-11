// nova_v2/code/graph/kv_pool.h — 共享 KV 块池（paged 非连续寻址，第 2 步；无跨会话共享）
//
// ============================ 它解决什么问题 ============================
// 每个会话的 KV 不再"预切一大片连续显存"，而是从**全卡共享的块池**里按块分配。
// 这样 N 个会话的总显存 = L*NB*KB*KVW（与逐会话切片**一个字节都不差**），
// 但地址不再连续 —— attention 必须按块表跳着读。第 3 步的会话共享 + COW 才能在此基础上做。
//
// ============================ 显存布局（k 与 v 分池，等价于把 [2] 维拆开） ============================
//     pool : [layer][NB][KB][NKV][HD]
//     NB   = kMaxSessions * nblocks      全局物理块数         (kMaxSessions=4, nblocks=ctx/KB)
//     KB   = kKvBlock                    每块容纳的 token 数   (16)
//     KVW  = NKV * HD                    每 token 的 float 数  (1.5B: 2*128 = 256)
//   地址 = pool + ((layer * NB + phys) * KB + slot) * KVW
//          slot = pos & (KB-1)            ← KB 是 2 的幂，所以用掩码不用取模
//          phys = table[pos >> kKvBlockShift]
//
// ============================ 核心参数（改之前先看清楚） ============================
//   kKvBlock      ∈ {8,16,32,64}，必须是 2 的幂（寻址靠移位/掩码）；
//                 改大 → 块内浪费变大（最后一个块用不满）；改小 → 块表变长、attention 跳更多次。
//   kMaxSessions  并发会话数，决定池的总量（4 × 2048 token × 28 层 ≈ 448 MiB）。
//   kMaxCtx       单会话上下文上限，必须是 kKvBlock 的整数倍，否则池会多出尾巴（有 static_assert 兜）。
//   table[]       每会话一张 int32 表，长度 = nblocks，逻辑块 → 物理块。
//                 第 1/2 步是恒等映射（table[b] = slice*nblocks + b），第 3 步才出现共享。
//
// ============================ 指针生命周期（谁分配、谁释放、活多久） ============================
//   · k / v        指向 DeviceArena 里的池基址 + 本切片偏移。**活到 arena 析构**；
//                  会话析构不释放它们（池是引擎级共享资源）。
//   · table_dev    指向 arena 里的块表副本（每会话一段）。写入用 cudaMemcpyAsync（H2D），
//                  生命周期同 arena；COW 时由 Engine::cow_guard 更新。
//   · table        host 侧镜像（std::vector），只用于调试/断言，kernel 不读它。
//
// ============================ COW（第 3 步，当前只有骨架） ============================
//   need_copy_from / need_copy_blk 是**一次性**信号：cow_slice() 判定"这个块被共享，
//   写前要复制"，把源块号记在这里；Engine::cow_guard 拷完必须把它们清成 -1，
//   否则下一次写入会重复拷贝（浪费带宽且可能覆盖新数据）。
//
// ============================ 边界检查 ============================
//   · `(pos >> kKvBlockShift)` 必须 < nblocks：由 kv_nblocks() = kMaxCtx/kKvBlock 保证，
//     而 pos < kMaxCtx 由调用方（Engine::decode / prefill）保证。
//   · 池满时 cow_slice 返回 -2，cow_guard 直接 abort —— 宁可崩也不静默写坏别人的 KV。
//
// ============================ 硬件差异 ============================
//   本文件是 **CUDA 线**的（共享块池 + 块表）。SYCL 线目前用**连续 KV**
//   （code/model/sycl/runner.cpp 的 kvk_/kvv_，布局 [L][ctx][NKV*HD]），没有 paged 寻址 ——
//   这是有意的里程碑切割（docs/SYCL_PORT.md §12.2），不要把两边的"KV 布局"当成同一个东西。
#pragma once
// 第 1 步的恒等映射（table[b] = slice*nblocks + b）继续保留 —— 每片仍然独占，
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdio>
#include "graph/kv_pool_cfg.h"

namespace nova {

struct KvSlice {
    int      need_copy_from = -1;   // COW 待拷贝源（一次性）
    int      need_copy_blk  = -1;
    int      id = -1;
    float*   k = nullptr;
    float*   v = nullptr;
    int64_t  n_past = 0;
    int      nblocks = 0;
    std::vector<int32_t> table;    // 逻辑块 -> 物理块（恒等映射 + 切片基址）
    int32_t*             table_dev = nullptr;   // 同一张表在设备上的副本
};

class KvPool {
public:
    void init(int n_slices, int64_t ctx) {
        ctx_ = ctx;
        nblocks_ = (int)((ctx + kKvBlock - 1) / kKvBlock);
        slices_.clear();
        slices_.reserve((size_t)n_slices);   // 指针会被 Session 长期持有，禁止搬迁
        free_.clear();
        for (int i = n_slices - 1; i >= 0; --i) free_.push_back(i);   // 先分配 0 号

        nb_total_ = n_slices * nblocks_;
        refcnt_.assign((size_t)nb_total_, 0);
        free_blocks_.clear();
        free_blocks_.reserve((size_t)nb_total_);
        for (int b = nb_total_ - 1; b >= 0; --b) free_blocks_.push_back(b);
        cow_count_ = 0;
    }

    // 设备侧块表总基址（长度 n_slices * nblocks，按 slice 取偏移）
    void set_dev(int32_t* base) { dev_ = base; }

    // 取一个独占切片：块表做恒等映射，物理块号 = slice*nblocks + b
    KvSlice* acquire() {
        if (free_.empty()) return nullptr;
        const int id = free_.back();
        free_.pop_back();
        slices_.push_back(KvSlice{});
        KvSlice* s = &slices_.back();
        s->id = id;
        s->n_past = 0;
        s->nblocks = nblocks_;
        s->table.resize((size_t)nblocks_);
        for (int b = 0; b < nblocks_; ++b) s->table[(size_t)b] = (int32_t)(id * nblocks_ + b);
        s->table_dev = dev_ ? dev_ + (size_t)id * (size_t)nblocks_ : nullptr;
        // 这些块被本切片占用：refcnt=1，并从空闲表摘掉
        if (refcnt_.empty()) {   // 还没有 init 过：先按当前规模初始化账本
            nb_total_ = (int)slices_.size() * nblocks_ > 0 ? nb_total_ : 0;
        }
        for (int b = 0; b < nblocks_ && (size_t)(id * nblocks_ + b) < refcnt_.size(); ++b) {
            const int phys = s->table[(size_t)b];
            refcnt_[(size_t)phys] = 1;
            auto it = std::find(free_blocks_.begin(), free_blocks_.end(), phys);
            if (it != free_blocks_.end()) free_blocks_.erase(it);
        }
        return s;
    }

    // ---- 块级分配 / 引用计数 ----
    int alloc_block() {
        if (free_blocks_.empty()) return -1;
        const int b = free_blocks_.back();
        free_blocks_.pop_back();
        refcnt_[(size_t)b] = 1;
        return b;
    }
    void free_block(int b) {
        if (b < 0 || b >= nb_total_) return;
        if (--refcnt_[(size_t)b] <= 0) { refcnt_[(size_t)b] = 0; free_blocks_.push_back(b); }
    }
    void ref_block(int b) { if (b >= 0 && b < nb_total_) ++refcnt_[(size_t)b]; }
    int  refcnt(int b) const { return (b >= 0 && b < nb_total_) ? refcnt_[(size_t)b] : 0; }
    int  nb_total() const { return nb_total_; }

    // 挂载：把 src 的前 k 个逻辑块指向 dst（引用同一物理块）
    void attach(KvSlice& dst, const KvSlice& src, int k) {
        for (int i = 0; i < k; ++i) {
            dst.table[(size_t)i] = src.table[(size_t)i];
            ref_block(src.table[(size_t)i]);
            dst.n_past = (int64_t)k * kKvBlock;
        }
    }

    // 按「token 数」挂载前缀：未对齐的尾块也会被共享（这是 COW 的真实触发场景）
    void attach_prefix(KvSlice& dst, const KvSlice& src, int64_t n_tokens) {
        const int nb = (int)((n_tokens + kKvBlock - 1) / kKvBlock);
        for (int i = 0; i < nb && i < (int)src.table.size(); ++i) {
            dst.table[(size_t)i] = src.table[(size_t)i];
            ref_block(src.table[(size_t)i]);
        }
        dst.n_past = n_tokens;
    }
    int cow_count() const { return cow_count_; }
    int free_blocks() const { return (int)free_blocks_.size(); }

    // COW：pos 所在的逻辑块若被共享，则复制一份独占块，返回新物理块号；无需复制返回 -1
    int cow_slice(KvSlice& s, int64_t pos) {
        const int lb = kv_block_index(pos);
        if (lb < 0 || lb >= (int)s.table.size()) return -1;
        const int phys = s.table[(size_t)lb];
        if (refcnt(phys) <= 1) return -1;
        const int nb = alloc_block();
        if (nb < 0) return -2;                    // 池满
        ++cow_count_;
        s.need_copy_from = phys;                  // 由调用方做设备拷贝
        s.need_copy_blk = nb;
        s.table[(size_t)lb] = nb;
        free_block(phys);
        return nb;
    }

    void release(KvSlice* s) {
        if (!s) return;
        for (int32_t b : s->table) free_block(b);   // 归还该会话引用的所有块
        free_.push_back(s->id);
        for (auto it = slices_.begin(); it != slices_.end(); ++it)
            if (it->id == s->id) { slices_.erase(it); return; }
    }

    // 逻辑位置 -> (物理块号, 槽位)
    int32_t phys_block(const KvSlice& s, int64_t pos) const {
        return s.table[(size_t)kv_block_index(pos)];
    }
    int slot(int64_t pos) const { return kv_slot_index(pos); }

    int      nblocks() const { return nblocks_; }
    int64_t  ctx() const { return ctx_; }
    int      used() const { return (int)slices_.size(); }
    int      free_slices() const { return (int)free_.size(); }

private:
    int64_t  ctx_ = 0;
    int      nblocks_ = 0;
    int32_t* dev_ = nullptr;
    std::vector<KvSlice> slices_;
    std::vector<int>     free_;
    std::vector<int>     free_blocks_;
    std::vector<int>     refcnt_;
    int                  nb_total_ = 0;
    int                  cow_count_ = 0;
};

inline KvPool& kv_pool() { static KvPool p; return p; }

} // namespace nova
