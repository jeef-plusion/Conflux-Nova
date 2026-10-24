// 计算侧前端 3/5：Token 向量 → 文本
#pragma once
#include <vector>
#include "map/qwen2.5/tokenizer.h"

namespace nova {

inline std::string tokens_to_text(const Tokenizer& tk, const std::vector<int>& ids) {
    return tk.decode(ids);
}

} // namespace nova
