// code/kernel/1080ti/kernel_api.h — 计算侧唯一入口（形状全是编译期常量）
#pragma once
#include "nova/kernel.h"
#include "nova/model.h"

namespace nova {
using KernelApi = Kernel;   // Kernel::q_proj_n / attn_decode / attn_prefill ...
}
