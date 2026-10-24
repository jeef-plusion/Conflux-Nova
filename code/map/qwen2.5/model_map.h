// code/map/qwen2.5/model_map.h — 模型清单（Chat 模板与会话元数据都在这里）
#pragma once
#include <cstdio>
#include <string>

namespace nova {

struct ModelCard {
    const char* name;
    const char* size;
    const char* im_start;   // ChatML 起止标记
    const char* im_end;
    const char* system;
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
    {"DeepSeek-R1-Distill-Llama-8B",     "8B",   "<|im_start|>", "<|im_end|>",
     "You are a helpful assistant."},
};
inline constexpr int kCardCount = sizeof(kCards) / sizeof(kCards[0]);

} // namespace nova
