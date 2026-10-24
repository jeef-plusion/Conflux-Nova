// 计算侧前端 2/5：模板串 → Token 向量（ChatML 特殊标记直接按 id 落，其余走 BPE）
#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include "map/qwen2.5/tokenizer.h"

namespace nova {

inline std::vector<int> encode_chat(const Tokenizer& tk, const std::string& text) {
    static const std::string kStart = "<|im_start|>";
    static const std::string kEnd   = "<|im_end|>";
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        const size_t p1 = text.find(kStart, i);
        const size_t p2 = text.find(kEnd, i);
        const size_t p  = std::min(p1, p2);
        if (p == std::string::npos) {
            const auto v = tk.encode(text.substr(i));
            ids.insert(ids.end(), v.begin(), v.end());
            break;
        }
        if (p > i) {
            const auto v = tk.encode(text.substr(i, p - i));
            ids.insert(ids.end(), v.begin(), v.end());
        }
        if (p == p1) { ids.push_back(tk.id(kStart)); i = p + kStart.size(); }
        else         { ids.push_back(tk.id(kEnd));   i = p + kEnd.size(); }
    }
    return ids;
}

} // namespace nova
