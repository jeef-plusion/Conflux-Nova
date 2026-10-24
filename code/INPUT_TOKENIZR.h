// 阶段 2/6：文本 → token（词表在前端）；越界则丢最老的一半并重算 → 调用 INPUT_START
#pragma once
#include "session.h"
#include "INPUT_START.h"

namespace nova {

inline std::vector<int> encode_chat(const Tokenizer& tk, const std::string& text) {
    // 先按所有已知特殊标记切分（ChatML + DeepSeek）
    const char* marks[] = {"<|im_start|>", "<|im_end|>", "<ï½Userï½>", "<ï½Assistantï½>", "<ï½endâofâsentenceï½>"};
    {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            size_t best = std::string::npos; const char* bm = nullptr;
            for (const char* m : marks) {
                const size_t p = text.find(m, i);
                if (p != std::string::npos && (best == std::string::npos || p < best)) { best = p; bm = m; }
            }
            if (best == std::string::npos) { auto v = tk.encode(text.substr(i)); out.insert(out.end(), v.begin(), v.end()); break; }
            if (best > i) { auto v = tk.encode(text.substr(i, best - i)); out.insert(out.end(), v.begin(), v.end()); }
            const int id = tk.id(bm);
            if (id >= 0) out.push_back(id);
            i = best + std::string(bm).size();
        }
        return out;
    }

    static const std::string kStart = "<|im_start|>";
    static const std::string kEnd   = "<|im_end|>";
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        const size_t p1 = text.find(kStart, i);
        const size_t p2 = text.find(kEnd, i);
        const size_t p  = (p1 < p2) ? p1 : p2;
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

inline void INPUT_TOKENIZR(Session* s) {
    Tokenizer& tk = *s->tk;
    s->prompt_ids = encode_chat(tk, s->delta);

    // 超限：放弃最老的一半历史，整段重算（绝不向系统借内存）
    if (s->pos0 + (int64_t)s->prompt_ids.size() > kMaxCtx) {
        const size_t keep_from = s->history.size() / 2;
        s->history.erase(s->history.begin(), s->history.begin() + keep_from);
        s->kv_valid = false;
        s->pos0 = 0;
        s->delta = render_full(*s->card, s->history, s->sys);
        s->prompt_ids = encode_chat(tk, s->delta);
        std::printf("\n[CTX] 滑窗截断：保留 %zu 条消息，重算 %zu token\n",
                    s->history.size(), s->prompt_ids.size());
    }

    INPUT_START(s);
}

} // namespace nova
