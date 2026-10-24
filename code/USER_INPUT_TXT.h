// 计算侧前端 1/5：文本 → ChatML 模板串
#pragma once
#include <string>
#include "map/qwen2.5/model_map.h"

namespace nova {

inline std::string build_prompt(const ModelCard& c, const std::string& user) {
    std::string s;
    s += c.im_start; s += "system\n";  s += c.system; s += c.im_end; s += "\n";
    s += c.im_start; s += "user\n";    s += user;     s += c.im_end; s += "\n";
    s += c.im_start; s += "assistant\n";
    return s;
}

} // namespace nova
