// nova_v2/model/select.h — 一个二进制 = 一个模型。模具在编译期焊死。
// 构建时必须点名：-DNOVA_MODEL_QWEN25_7B / _R1_QWEN7B / _QWEN25_1P5B / _R1_QWEN1P5B / _R1_LLAMA8B
#pragma once
#include "nova/model/matrix.h"

#if defined(NOVA_MODEL_QWEN25_7B)
using ActiveModel = nova::Qwen25_7B;
inline constexpr const char* kModelName = "Qwen2.5-7B-Instruct";
inline constexpr const char* kModelAliases[] = {"Qwen2.5-7B-Instruct", "Qwen2.5-7B"};
#elif defined(NOVA_MODEL_R1_QWEN7B)
inline constexpr const char* kModelName = "DeepSeek-R1-Distill-Qwen-7B";
inline constexpr const char* kModelAliases[] = {"DeepSeek-R1-Distill-Qwen-7B", "R1-Distill-Qwen-7B"};
using ActiveModel = nova::R1_Qwen7B;
#elif defined(NOVA_MODEL_QWEN25_1P5B)
inline constexpr const char* kModelName = "Qwen2.5-1.5B-Instruct";
inline constexpr const char* kModelAliases[] = {"Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B"};
using ActiveModel = nova::Qwen25_1p5B;
#elif defined(NOVA_MODEL_R1_QWEN1P5B)
inline constexpr const char* kModelName = "DeepSeek-R1-Distill-Qwen-1.5B";
inline constexpr const char* kModelAliases[] = {"DeepSeek-R1-Distill-Qwen-1.5B", "R1-Distill-Qwen-1.5B"};
using ActiveModel = nova::R1_Qwen1p5B;
#elif defined(NOVA_MODEL_R1_LLAMA8B)
inline constexpr const char* kModelName = "DeepSeek-R1-Distill-Llama-8B";
inline constexpr const char* kModelAliases[] = {"DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B"};
using ActiveModel = nova::R1_Llama8B;
#else
#error "no model selected: pass -DNOVA_MODEL_QWEN25_7B (or another)"
#endif
