// 阶段 3/6：计算侧针脚（只吃 token id + 本会话 KV/工作区）→ 每 token 自己触发 OUT_TKZ
#pragma once
#include <chrono>
#include "session.h"
#include "OUT_TKZ.h"

namespace nova {

inline void INPUT_START(Session* s) {
    Engine& e = *s->engine;
    const int im_end = s->tk->id(s->card->im_end, 151645);
    const int n_new  = (int)s->prompt_ids.size();

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    e.prefill(s->ws, s->prompt_ids.data(), n_new, s->kvk, s->kvv, s->pos0);
    const auto t1 = clk::now();

    int64_t p = s->pos0 + n_new;
    s->out_ids.reserve(s->max_new);
    int tok = e.sample_last(s->ws);
    while ((int)s->out_ids.size() < s->max_new && tok != im_end) {
        s->out_ids.push_back(tok);
        OUT_TKZ(s);
        tok = e.decode(s->ws, tok, s->kvv ? s->kvk : s->kvk, s->kvv, p);
        ++p;
    }
    const auto t2 = clk::now();

    s->prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s->decode_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    s->n_past     = p;
    s->kv_valid   = true;
    OUT_OUT_DONE(s);
}

} // namespace nova
