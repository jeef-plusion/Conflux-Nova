// tests/kv_block_test.cpp — KV 块哈希的宿主自测（不需要 GPU）
#include <cstdio>
#include <vector>
#include "graph/kv_block.h"

int main() {
    const std::vector<int> a{151644, 8948, 198, 2610};
    const std::vector<int> b{151644, 8948, 198, 2611};
    const uint64_t ha = nova::kv_hash_ids(a);
    const uint64_t hb = nova::kv_hash_ids(b);
    const uint64_t ha2 = nova::kv_hash_ids(a);

    std::printf("[kv] a      = %s\n", nova::kv_hash_hex(ha).c_str());
    std::printf("[kv] b      = %s\n", nova::kv_hash_hex(hb).c_str());
    std::printf("[kv] block  = %s\n", nova::kv_block_id(ha, hb).c_str());

    int rc = 0;
    if (ha != ha2) { std::printf("[kv] FAIL: 非确定性\n"); rc = 1; }
    if (ha == hb)  { std::printf("[kv] FAIL: 不同输入撞了\n"); rc = 1; }
    if (nova::kv_hash_hex(ha).size() != 16) { std::printf("[kv] FAIL: 不是 16 个 hex 字符\n"); rc = 1; }
    std::printf(rc ? "KV BLOCK FAIL\n" : "KV BLOCK PASS\n");
    return rc;
}
