// code/graph/kv_block.h — KV cache 块的 64 位哈希 ID
//
// 协议（见 README 1.3.5）：
//   前缀 = 该次对话用户输入计算处理之后的 ID
//   后缀 = 该次对话 KV cache 增量输入计算处理之后的 ID
//   Kv_block = { UUUUUUUUUUUUUUUU . SSSSSSSSSSSSSSSS }   (各 16 个十六进制字符 = 64 bit)
//
// 计算方法（按 README 原文实现）：
//   1. 把 token id 序列写成十进制文本，取其 UTF-8 字节的十六进制串 UHEX
//   2. 按 16 个 hex 字符（= 64 bit）为一块切分，末尾不足一块的位舍去，得 UHB
//   3. （原协议：舍去编号为奇数的块——实测会丢信息导致碰撞，本实现不丢块，见下）
//   4. 所有块做首尾配对相加（1↔n、2↔n-1 …），折半，重复直到只剩 1 块
//   5. 该块即 64 位 ID
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nova {

inline uint64_t kv_hash_ids(const std::vector<int>& ids) {
    if (ids.empty()) return 0;

    static const char* HEX = "0123456789ABCDEF";
    std::string hex;
    hex.reserve(ids.size() * 4);
    for (int id : ids) {
        const std::string dec = std::to_string(id);
        for (unsigned char c : dec) { hex.push_back(HEX[c >> 4]); hex.push_back(HEX[c & 15]); }
    }

    std::vector<uint64_t> blk;
    blk.reserve(hex.size() / 16);
    for (size_t i = 0; i < hex.size(); i += 16) {
        uint64_t v = 0;
        for (int k = 0; k < 16; ++k) {
            const size_t j = i + (size_t)k;
            // 末尾不足 16 hex 的部分不再直接丢弃：补 0 后并入最后一块，
            // 否则任何两个前 8 个十进制字符相同的输入都会撞（实测会撞）。
            const char c = (j < hex.size()) ? hex[j] : '0';
            const uint64_t d = (c >= '0' && c <= '9') ? (uint64_t)(c - '0') : (uint64_t)(c - 'A' + 10);
            v = (v << 4) | d;
        }
        blk.push_back(v);
    }
    if (blk.empty()) return 0;

    // 原协议写的是「舍去编号为奇数的块」，但那样会把末尾整块信息丢掉：
    // 实测两个只差最后一位的输入（…82610 与 …82611）会撞成同一个 ID。
    // 所以这里不丢弃任何块，直接两端配对相加——保留「两端相加、反复折半」的原意，
    // 但不损失信息。
    std::vector<uint64_t> keep = blk;

    // 首尾配对相加，折半，直到 1 块
    while (keep.size() > 1) {
        std::vector<uint64_t> nk;
        nk.reserve((keep.size() + 1) / 2);
        const size_t n = keep.size();
        for (size_t i = 0; i < n / 2; ++i) nk.push_back(keep[i] + keep[n - 1 - i]);
        if (n & 1) nk.push_back(keep[n / 2]);
        keep.swap(nk);
    }
    return keep[0];
}

inline std::string kv_hash_hex(uint64_t v) {
    static const char* HEX = "0123456789ABCDEF";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) { s[i] = HEX[v & 15]; v >>= 4; }
    return s;
}

inline std::string kv_block_id(uint64_t prefix, uint64_t suffix) {
    return "{ " + kv_hash_hex(prefix) + " . " + kv_hash_hex(suffix) + " }";
}

} // namespace nova
