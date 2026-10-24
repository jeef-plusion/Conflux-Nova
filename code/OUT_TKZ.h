// 阶段 4/6：token → 字节 → 只切完整 UTF-8 前缀 → 直接调用 OUT_chat
#pragma once
#include "session.h"
#include "OUT_chat.h"

namespace nova {

inline void OUT_TKZ(Session* s) {
    const int id = s->out_ids.back();
    s->pending += s->tk->decode(std::vector<int>{id});

    size_t i = 0, last = 0;
    while (i < s->pending.size()) {
        const unsigned char c = (unsigned char)s->pending[i];
        const int n = (c < 0x80) ? 1 : ((c >> 5) == 6) ? 2 : ((c >> 4) == 14) ? 3 : ((c >> 3) == 30) ? 4 : 1;
        if (i + (size_t)n > s->pending.size()) break;
        i += (size_t)n;
        last = i;
    }
    if (last == 0) return;

    const std::string chunk = s->pending.substr(0, last);
    s->pending.erase(0, last);
    OUT_chat(s, chunk);
}

} // namespace nova
