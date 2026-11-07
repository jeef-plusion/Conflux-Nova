// code/graph/kv_pool.h — 共享 KV 块池（第 2 步：非连续寻址 / Paged Attention，仍无共享）
//
// 显存布局（两个池各一份，k 与 v 分池存放，等价于 [2] 维拆开）：
//       pool: [layer][NB][KB][NKV][HD]
//       NB  = kMaxSessions * nblocks        // 全局物理块数
//       KB  = kKvBlock（16）                 // 每块 token 数
//       KVW = NKV * HD                       // 每 token 的 float 数
// 地址 = pool + ((layer * NB + phys) * KB + slot) * KVW
//        slot = pos & (KB-1)，phys = table[pos >> kKvBlockShift]
//
// 总量与改造前的逐会话切片完全相同（L*NB*KB*KVW == kMaxSessions*span），
// 所以分配大小一个字节都没变，变的只有寻址：每会话一张块表，块表决定物理块号。
//
// 第 1 步的恒等映射（table[b] = slice*nblocks + b）继续保留 —— 每片仍然独占，
// 只是切片内的地址不再连续，attention 要按块跳。第 3 步才做共享挂载 + COW。
#pragma once
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
