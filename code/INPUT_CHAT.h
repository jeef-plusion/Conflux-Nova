// 阶段 1/6：收下这一轮的 user 文本，拼出「增量」ChatML → 直接调用 INPUT_TOKENIZR
#pragma once
#include "session.h"
#include "INPUT_TOKENIZR.h"

namespace nova {

inline void INPUT_CHAT(Session* s, const char* text) {
    s->user_text = text ? text : "";
    s->history.push_back({"user", s->user_text});

    s->out_ids.clear();
    s->reply.clear();
    s->pending.clear();
    s->prompt_ids.clear();

    const ModelCard& c = *s->card;
    if (!s->kv_valid) {
        s->delta = render_full(c, s->history, s->sys);   // 首轮 / 截断后：全量
        s->pos0  = 0;
    } else {
        // 上一轮没写 im_end（生成到 im_end 就停了），这里补上再开新回合
        s->delta = std::string(c.im_end) + "\n" + c.im_start + "user\n" + s->user_text
                 + c.im_end + "\n" + c.im_start + "assistant\n";
        s->pos0 = s->n_past;
    }

    INPUT_TOKENIZR(s);
}

} // namespace nova
