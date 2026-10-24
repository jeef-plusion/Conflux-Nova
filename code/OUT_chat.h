// 阶段 5/6：去特殊标记 → 直接调用 OUT_OUT
#pragma once
#include <string>
#include "session.h"
#include "OUT_OUT.h"

namespace nova {

inline std::string clean_reply(std::string t) {
    for (const char* m : {"<|im_end|>", "<|im_start|>", "<|endoftext|>"}) {
        const std::string pat(m);
        for (size_t p; (p = t.find(pat)) != std::string::npos; ) t.erase(p, pat.size());
    }
    while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
    while (!t.empty() && (t.front() == '\n' || t.front() == ' ')) t.erase(t.begin());
    return t;
}

inline void OUT_chat(Session* s, std::string piece) {
    for (const char* m : {"<|im_end|>", "<|im_start|>", "<|endoftext|>"}) {
        const std::string pat(m);
        for (size_t p; (p = piece.find(pat)) != std::string::npos; ) piece.erase(p, pat.size());
    }
    if (piece.empty()) return;
    s->reply += piece;
    OUT_OUT(s, piece);
}

} // namespace nova
