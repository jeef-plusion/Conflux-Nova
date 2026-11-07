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
#include <cstdio>
#include "graph/kv_pool_cfg.h"

namespace nova {

struct KvSlice {
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
        return s;
    }

    void release(KvSlice* s) {
        if (!s) return;
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
};

inline KvPool& kv_pool() { static KvPool p; return p; }

} // namespace nova
