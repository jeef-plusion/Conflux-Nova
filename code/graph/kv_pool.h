// code/graph/kv_pool.h — 共享 KV 块池（第 1 步：独占映射，refcnt 恒为 1）
//
// 布局：pool 由 kMaxSessions 个「切片」组成，每片 = 一个会话的完整 KV：
//       [layer][token][2][NKV][HD]
// 切片 = 连续的 nblocks 个物理块（每块 KB 个 token），所以：
//       物理块号 phys = slice * nblocks + 逻辑块号
//       切片内地址   = layer * ctx * KVW + (b * KB + slot) * KVW
// 与改造前的逐会话切片布局完全一致 —— 这是「零语义变更」的前提。
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
    std::vector<int32_t> table;   // 逻辑块 -> 物理块（第 1 步为恒等映射 + 切片基址）
};

class KvPool {
public:
    void init(int n_slices, int64_t ctx) {
        ctx_ = ctx;
        nblocks_ = (int)((ctx + kKvBlock - 1) / kKvBlock);
        slices_.clear();
        free_.clear();
        for (int i = n_slices - 1; i >= 0; --i) free_.push_back(i);   // 先分配 0 号
    }

    // 取一个独占切片：块表做恒等映射，向后兼容连续布局
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
    int64_t ctx_ = 0;
    int     nblocks_ = 0;
    std::vector<KvSlice> slices_;
    std::vector<int>     free_;
};

inline KvPool& kv_pool() { static KvPool p; return p; }

} // namespace nova
