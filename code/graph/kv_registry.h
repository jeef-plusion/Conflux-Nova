// code/graph/kv_registry.h — KV cache 块的登记表
//
// 设计原则（见 README 1.3.5）：
//   ID 是地址，只能由确定性输入构成；模型输出是内容，不进地址。
//   前缀 = 本对话首轮用户输入的哈希
//   后缀 = 本轮用户输入的哈希，异或轮次序号（同一句话问两次不会撞）
//   载荷 = KV 区间 [pos0, pos1) + 会话号 + 轮次
#pragma once
#include <cstdint>
#include <vector>
#include <cstdio>
#include "graph/kv_block.h"

namespace nova {

struct KvBlock {
    uint64_t prefix = 0;
    uint64_t suffix = 0;
    int      session = -1;
    int      turn = 0;
    int64_t  pos0 = 0;
    int64_t  pos1 = 0;
    bool     alive = true;
};

class BlockRegistry {
public:
    void set_capacity(size_t n) { cap_ = n ? n : 1; evict(); }
    size_t capacity() const { return cap_; }

    // 同 (prefix, suffix) 视为同一块：旧条目先失效，再插入新条目
    void put(const KvBlock& b) {
        for (auto& x : v_) if (x.alive && x.prefix == b.prefix && x.suffix == b.suffix) x.alive = false;
        v_.push_back(b);
        evict();
    }

    const KvBlock* find(uint64_t prefix, uint64_t suffix) const {
        for (const auto& x : v_) if (x.alive && x.prefix == prefix && x.suffix == suffix) return &x;
        return nullptr;
    }

    int live() const { int n = 0; for (const auto& x : v_) if (x.alive) ++n; return n; }
    int total() const { return (int)v_.size(); }

    void dump() const {
        std::printf("[KV] registry: live=%d total=%d cap=%zu\n", live(), total(), cap_);
        for (const auto& x : v_) {
            if (!x.alive) continue;
            std::printf("[KV]   { %s . %s }  session=%d turn=%d kv=[%lld,%lld)\n",
                        kv_hash_hex(x.prefix).c_str(), kv_hash_hex(x.suffix).c_str(),
                        x.session, x.turn, (long long)x.pos0, (long long)x.pos1);
        }
    }

private:
    void evict() {
        // 超出容量：从最老的活块开始失效
        while ((size_t)live() > cap_) {
            for (auto& x : v_) if (x.alive) { x.alive = false; break; }
        }
    }

    std::vector<KvBlock> v_;
    size_t cap_ = 64;
};

inline BlockRegistry& blocks() { static BlockRegistry r; return r; }

} // namespace nova
