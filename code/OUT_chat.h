// 计算侧前端 4/5：输出清洗（去掉特殊标记、行尾空白）
#pragma once
#include <string>

namespace nova {

inline std::string clean_reply(std::string s) {
    for (const char* m : {"<|im_end|>", "<|im_start|>", "<|endoftext|>"}) {
        for (size_t p; (p = s.find(m)) != std::string::npos; ) s.erase(p, std::string(m).size());
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r')) s.pop_back();
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.erase(s.begin());
    return s;
}

} // namespace nova
