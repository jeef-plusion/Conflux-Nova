// nova_v2/loader/validate.h — 特征签名校验：文件自描述 vs 编译期模具
#pragma once
#include <cstdint>
#include <string>
#include "nova/loader/gguf.h"

namespace nova {

struct Verification {
    uint64_t    weight_bytes = 0;
    uint32_t    n_weight = 0;
    uint32_t    n_norm = 0;
    std::string name;          // general.name（或文件名）
    bool        variant = false; // 签名一致但名字不在官方白名单
};

// 读 metadata → 比对签名 → 张量表/量化路径/预算；不匹配直接 abort
Verification verify_model(const GgufFile& g);

} // namespace nova
