// 计算侧前端 5/5：最终输出
#pragma once
#include <cstdio>
#include <string>

namespace nova {

inline void emit(const std::string& text, double prefill_ms, double decode_ms, int n_new) {
    std::printf("\n[ANSWER] %s\n\n", text.c_str());
    std::printf("[PERF] prefill %.1f ms  decode %.1f ms  %d tokens  (%.2f tok/s)\n",
                prefill_ms, decode_ms, n_new, n_new * 1000.0 / (decode_ms > 0 ? decode_ms : 1.0));
    std::fflush(stdout);
}

} // namespace nova
