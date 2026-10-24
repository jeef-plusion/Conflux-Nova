// code/map/qwen2.5/model_map.h — 模型清单（Chat 模板与会话元数据都在这里）
#pragma once
#include <cstdio>
#include <string>

namespace nova {

struct ModelCard {
    const char* name;
    const char* size;
    const char* im_start;   // 用户回合标记（ChatML: <|im_start|> / DS: <｜User｜>）
    const char* im_end;
    const char* system;
    const char* asst = nullptr;   // 非空 = DeepSeek 风格，助手回合用这个标记
};

inline constexpr ModelCard kCards[] = {
    {"Qwen2.5-7B-Instruct",              "7B",   "<|im_start|>", "<|im_end|>",
     "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"Qwen2.5-1.5B-Instruct",            "1.5B", "<|im_start|>", "<|im_end|>",
     "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"DeepSeek-R1-Distill-Qwen-7B",      "7B",   "<|im_start|>", "<|im_end|>",
     "You are a helpful assistant."},
    {"DeepSeek-R1-Distill-Qwen-1.5B",    "1.5B", "<|im_start|>", "<|im_end|>",
     "You are a helpful assistant."},
    {"DeepSeek-R1-Distill-Qwen-1.5B-Abliterated", "1.5B", "\xef\xbd\x9c" "User" "\xef\xbd\x9c", "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c", "",
     "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c"},
    {"DeepSeek-R1-Distill-Llama-8B",     "8B",   "<|im_start|>", "<|im_end|>",
     "You are a helpful assistant."},
};
inline constexpr int kCardCount = sizeof(kCards) / sizeof(kCards[0]);

} // namespace nova
