// 阶段 6/6：写本会话缓冲；收尾时加锁一次性落终端（防串台）
#pragma once
#include <cstdio>
#include "session.h"

namespace nova {

inline void OUT_OUT(Session* s, const std::string& piece) { s->obuf += piece; }

inline void OUT_OUT_DONE(Session* s) {
    const int n = (int)s->out_ids.size();
    const double tps = n * 1000.0 / (s->decode_ms > 0 ? s->decode_ms : 1.0);
    std::lock_guard<std::mutex> lk(console_mu());
    std::printf("\n===== [S%d] %s =====\n", s->id, s->user_text.c_str());
    std::printf("%s\n", s->obuf.c_str());
    std::printf("[S%d] prefill %.1f ms (%d new tok) | decode %.1f ms | %d tok | %.2f tok/s | n_past=%lld\n",
                s->id, s->prefill_ms, (int)s->prompt_ids.size(), s->decode_ms, n, tps, (long long)s->n_past);
    std::fflush(stdout);
    s->history.push_back({"assistant", s->reply});
}

} // namespace nova
