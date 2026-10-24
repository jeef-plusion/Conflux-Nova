# Intel Arc 140T (Core Ultra 7 255H) 后端技术调研

> 目标机型：联想小新 Pro 14 2025（Arrow Lake-H / Core Ultra 7 255H / Arc 140T / 32 GB LPDDR5x-8533 / Win11）
> 现状：nova_v2 为 CUDA sm_61 单体引擎，算子 = Q4_K GEMV(dp4a)、MMQ 瓦片、RMSNorm、RoPE、attention(decode split-K / prefill flash)、embed、quant(fp32→int8)。
> 本文只做调研，不改任何代码。文末附全部来源；查不到的条目明确写「未查证」。

## 0. 结论先行

| # | 结论 | 影响 |
|---|---|---|
| 1 | **SYCL（oneAPI DPC++）是首选**：`sycl::dp4a` / subgroup / local_accessor 与我们的 CUDA 写法一一对应（约 70% 结构可沿用）。Vulkan 降为备选（GLSL 翻译量大约一倍，约 40% 可沿用）；Level Zero 依赖最轻但 C API 手写描述符最费工。**「跨厂商」不构成选型理由**（见第 2 节修订说明）。 | 后端锁定 SYCL，Vulkan 仅作退路 |
| 2 | **int8 点积在这台机器上是硬件加速的**：驱动上报 `integerDotProduct4x8BitPackedSignedAccelerated = true`、`shaderInt8 = true`（两个不同驱动版本均如此）。 | Q4_K 的 dp4a 内层可 1:1 迁移 |
| 3 | **真正的天花板是内存带宽，不是算力**。理论 136.5 GB/s，实测有效 ~93–107 GB/s。7B Q4_K 解码上限 31.9 tok/s（理论）/ 23.3（按 100 GB/s），现实 13–17 tok/s。 | 优化目标是「少读一个字节」，不是「多算一条指令」 |
| 4 | **两个 Windows 驱动的硬限制必须先设计进去**：单次分配上限 3.99 GiB、单个 storage buffer 上限 1.0 GiB。7B 的 3.99 GiB 权重必须切片。 | 权重按层/按 ≤1 GiB 分块，不能一个大 buffer |
| 5 | **最大未知是 Intel 驱动的 int8 正确性**：已有 Arc A770 上 int8 dot 触发 device lost、以及特定驱动版本区间 [32.0.101.8509, 32.0.101.8860) 错编译的公开记录。 | 必须固定驱动版本 + 保留 fp16/fp32 回退 kernel |

---
## 1. 硬件事实

### 1.1 GPU 与算力
| 项目 | 数值 | 来源 |
|---|---|---|
| GPU | Intel Arc 140T（Xe2-LPG，Arrow Lake-H 核显） | [NanoReview 255H](https://nanoreview.net/en/cpu/intel-core-ultra-7-255h) |
| Xe 核心 | 8 | 同上（"Execution Units: 8" 即 Xe core 口径） |
| FP32 lane（着色器） | 1024（= 128 XVE × 8） | 同上 |
| 频率 | 300 MHz 基础 / **2250 MHz boost** | 同上 |
| FP32 峰值 | 4.6 TFLOPS（1024 × 2.25 GHz × 2） | 同上，与算式一致 |
| iGPU TGP | 35 W | 同上 |
| XMX（矩阵单元） | 8 个，int8 GEMM 实测 **41 TOPS @4096** | [lhl/intel-inference](https://github.com/lhl/intel-inference)（Lunar Lake 140V 实测） |
| dp4a 峰值（推算） | 1024 lane × 2.25 GHz = **2.3 T dp4a/s ≈ 9.2 TMAC/s int8** | 本文推算（非实测） |

> ⚠️ intel.com ARK 页面在本机网络不可达（`fetch failed`），上表的 Xe 核心数/频率来自二级来源；**上机第一件事是用 GPU-Z / `vulkaninfo` 复核 `maxComputeWorkGroupInvocations` 与设备名**。

### 1.2 显存策略（Windows / UMA）
| 项目 | 事实 | 来源 |
|---|---|---|
| 架构 | 核显与 CPU 共享同一块 LPDDR5x，**无独立 VRAM**，Vulkan 报 `INTEGRATED_GPU` | [gpuinfo #44273](http://vulkan.gpuinfo.org/displayreport.php?id=44273) |
| 单次分配上限 | `maxMemoryAllocationSize = 4294901760` ≈ **3.999 GiB** | 同上 |
| 单 buffer 上限 | `maxStorageBufferRange = 1073741820` ≈ **1.0 GiB** | 同上 |
| 共享内存上限（"能拿多少系统内存"） | Windows 侧由 WDDM 决定，任务管理器"共享 GPU 内存"通常约为物理内存一半 | **未查证**（intel.com / learn.microsoft.com 在本机不可达）；以目标机 `vkGetPhysicalDeviceMemoryProperties` 实测为准 |
| 工程结论 | 32 GB 机型上 7B Q4_K（3.99 GiB）+ KV + 激活**可以放下**，但权重必须拆成 ≥4 个 buffer（受 1 GiB 限制），且不能依赖单次 3.99 GiB 分配（余量仅 ~10 MB） | 本文结论 |

### 1.3 LPDDR5x-8533 真实可用带宽
| 项目 | 数值 | 说明 / 来源 |
|---|---|---|
| 理论带宽 | 8533 MT/s × 128 bit / 8 = **136.5 GB/s** | JEDEC 口径 + 双通道 128-bit；第三方亦给 137 GB/s（[canitrun](https://canitrun.dev/gpus/arc-140v-16/)） |
| 实测有效带宽（vector-add） | **93–107 GB/s** | [lhl/intel-inference](https://github.com/lhl/intel-inference)（同代 140V / LPDDR5x-8533 实测） |
| 实测 H2D / D2H / D2D 拷贝 | 45–54 GB/s | 同上（UMA 拷贝路径开销大，**不要用拷贝路径喂激活**） |
| 设计取值 | **100 GB/s（约 73%）** | 本文用于性能模型 |
| 风险 | 若该机实际是 **DDR5-5600 SO-DIMM**（部分资料如此描述），带宽降到 89.6 GB/s，全文数字 ×0.66 | [ZOL 产品页](https://nb.zol.com.cn/977/9779201_all.html)；**必须上机核对** `wmic memorychip` / CPU-Z |

### 1.4 与 GTX 1080 Ti 的对照（为什么不能照搬性能直觉）
| 指标 | GTX 1080 Ti (sm_61) | Arc 140T | 比值 |
|---|---|---|---|
| FP32 峰值 | 11.3 TFLOPS | 4.6 TFLOPS | 0.41× |
| 显存带宽 | 484 GB/s（GDDR5X 352-bit） | 136.5 GB/s | **0.28×** |
| 显存 | 11 GB 独占 | 32 GB 共享（单次分配 ≤3.99 GiB） | — |
| int8 dp4a | 有（`__dp4a`） | 有（`dotPacked4x8EXT`） | 1:1 |
| 解码定位 | 算力有余、带宽受限 | **带宽受限更严重** | 同模型 tok/s 先乘 0.28，再乘 iGPU 系数 |

> 结论：现有 kernel 的**算术结构**几乎不用改，**数据搬运结构**必须重做（缓存复用、预取深度、每字节权重摊到的 dp4a 数）。

---
## 2. 计算后端选型对比

> **修订（2026-10-24）**：初版把「跨厂商」当成 Vulkan 的主要优势，这是错的。
> SPIR-V 只保证**接口层**能提交，同一份 shader 在 Intel / AMD / Nvidia 上的代码生成路径完全不同，
> 所谓可移植最后必然分叉成三套调优。选型只看三条：**指令映射度、移植工作量、依赖重量**。

| 后端 | 指令映射度 | 移植工作量 | 依赖重量 | 结论 |
|---|---|---|---|---|
| **SYCL（oneAPI DPC++）** | 高：`sycl::dp4a` 官方 intrinsic、subgroup 与 warp 同构、`local_accessor` 等价 `__shared__` | **低**：现有 CUDA kernel 约 70% 结构可沿用 | 重：oneAPI 运行时（数百 MB）+ DPC++ 编译器 | **首选** |
| Level Zero | 高：Intel 原生底层 API | 高：C API，描述符全手写 | 轻：驱动自带 `ze_loader.dll` | 备选（想彻底去掉运行时） |
| Vulkan compute | 中：GLSL + 扩展，`dotPacked4x8EXT` 与 dp4a 对应但需翻译 | 高：约 40% 可沿用，descriptor set / pipeline barrier 全重写 | 中：系统自带 + glslc | 备选（oneAPI 装不起来时） |
| CPU AVX-VNNI | — | 低 | 无 | 对照与回退（Arrow Lake 无 AVX-512） |

### 为什么 SYCL 排第一

我们的 kernel 是逐行从 CUDA 语义写出来的，三件套在 SYCL 里都有对应：

| CUDA | SYCL |
|---|---|
| `__shfl_xor_sync(mask, v, 16)` | subgroup shuffle / `permute_group_by_xor` |
| `__shared__ float s[256]` | `sycl::local_accessor<float,1>` |
| `__dp4a(a, b, c)` | `sycl::dp4a(a, b, c)` |
| `__syncthreads()` | `item.barrier(fence_space::local_space)` |

同样的 GEMV，SYCL 版能保留主体结构，Vulkan 版要把这些全部翻译成 GLSL 并自己管描述符。

### 依赖代价（明确接受）

1. 多一个编译器（`icpx -fsycl`），构建脚本分叉；
2. 目标机需要 oneAPI 运行时（或用 redistributable 打包 DLL），不再是「一个 exe 拷过去就能跑」；
3. 锁死 Intel，将来要跑 AMD 核显需另写一套。

这三条都接受，因为 2.2.0 只服务小新 Pro 14 这一台机器。

## 3. 关键技术点核实（最重要）

### 3.1 `VK_KHR_shader_integer_dot_product` 与 `shaderInt8`
| 结论 | 证据 | 出处 |
|---|---|---|
| **暴露，且已进核心**：Vulkan 1.3 起该扩展被提升为核心特性，1.4 驱动必然具备 | 规范："Promoted to Vulkan 1.3" | [VK_KHR_shader_integer_dot_product](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_integer_dot_product.html) |
| **Windows 驱动实报为 true** | `shaderIntegerDotProduct=true`；`integerDotProduct4x8BitPackedSignedAccelerated=true`、`...UnsignedAccelerated=true`、`8BitSignedAccelerated=true`（驱动 101.8250，Vulkan 1.4.325） | [gpuinfo #44273](http://vulkan.gpuinfo.org/displayreport.php?id=44273) |
| **旧驱动同样为 true**（非新驱动才修好） | 驱动 101.5593 的报告里同类字段一致为 true | [gpuinfo #31555](http://vulkan.gpuinfo.org/displayreport.php?id=31555) |
| **`shaderInt8` 可用** | `shaderInt8 = true`（两个驱动版本一致）；但注意它只保证 int8 类型可用，**不保证算术按 8-bit 速率执行** | 同上 |
| 语义 | accelerated = 实现可用少量指令完成该点积（相对手工展开）；非 accelerated 只是能用，可能被拆开 | [VkPhysicalDeviceShaderIntegerDotProductProperties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderIntegerDotProductProperties.html) |

### 3.2 subgroup 尺寸
| 项目 | 实测值 | 出处 |
|---|---|---|
| 默认 subgroup | **32** | [gpuinfo #44273](http://vulkan.gpuinfo.org/displayreport.php?id=44273) |
| 支持范围 | `minSubgroupSize = 8`，`maxSubgroupSize = 32`（即 8 / 16 / 32 都可，需 `VK_EXT_subgroup_size_control` 指定） | 同上 |
| 每 workgroup 最大 subgroup 数 | 64 | 同上 |
| 社区实践 | llama.cpp Vulkan 在 Intel 上 `warp size: 32` | [llama.cpp #30692 日志](https://github.com/ggml-org/llama.cpp/issues/30692) |

**实操建议**：先按 **subgroup = 32，一个 subgroup 处理一行**，与现有 CUDA "一个 warp 一行" 完全同构，归约用 `subgroupAdd`。subgroup=16 在理论上能提高 EU 占用率（同一 EU 塞 2 个 subgroup 掩延迟），但会退化成两次归约 —— **哪个更快必须实测**（实验 E1），不要凭直觉选。

### 3.3 shared memory 与 workgroup 上限
| 项目 | Windows 驱动 | Linux Mesa（对照） | 出处 |
|---|---|---|---|
| `maxComputeSharedMemorySize` | **32768 B（32 KB）** | 49152 B（48 KB） | Win: [#44273](http://vulkan.gpuinfo.org/displayreport.php?id=44273)；Linux: [llama.cpp #30692 日志](https://github.com/ggml-org/llama.cpp/issues/30692) |
| `maxComputeWorkGroupInvocations` | 1024 | — | #44273 |
| `maxComputeWorkGroupSize` | 1024 / 1024 / 64 | — | 同上 |
| `maxComputeWorkGroupCount` | 65536³ | — | 同上 |
| `maxPushConstantsSize` | **256 B** | — | 同上 |

**对我们的直接后果**：现有 MMQ 瓦片 smem = `sw[64×9 int4]=9216 B` + `sx[32×256]=8192 B` + scale/sum ≈ **17.7 KB**，**能放进 32 KB**；但**不能**做"双缓冲 + 更大 token 瓦片"（那会超 32 KB，且 Windows 不给 48/64 KB）。→ 预填充阶段要么缩瓦片，要么走 cooperative matrix（见 3.5）。

### 3.4 fp16 存储 + fp32 累加
| 结论 | 证据 |
|---|---|
| ✅ 支持 | `shaderFloat16 = true`、`storageBuffer16BitAccess = true`、`uniformAndStorageBuffer16BitAccess = true`（[#44273](http://vulkan.gpuinfo.org/displayreport.php?id=44273)） |
| 用法 | Q4_K 的 `d`/`dmin` 直接用 `float16_t` 存/读，shader 内 `float(d)` 转 fp32 再累加；累加器一律 fp32（与 CUDA 版逐位对齐的关键） |
| 注意 | Vulkan 只保证 fp16 类型可算；`f16` 的乘加是否双倍速率取决于硬件（Xe2 上 fp16 与 fp32 同速为主，实测 bf16/fp16 GEMM ~22–24 TFLOPS vs fp32 3.9 TFLOPS，见 [lhl](https://github.com/lhl/intel-inference)） |

### 3.5 有没有"一对一"的 `__dp4a`？
| 问题 | 结论 |
|---|---|
| 有没有等价内建？ | **有**。GLSL 扩展 `GL_EXT_shader_integer_dot_product` 提供 `dotPacked4x8EXT(int,int)`，编译为 SPIR-V `OpSDot` + `PackedVectorFormat4x8Bit`。llama.cpp 的 Q4_K MMQ 就走这条路：`q_sum += dotPacked4x8EXT(cache_a[..], cache_b..)`（[mul_mmq_funcs.glsl](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl)） |
| 会不会被拆成多条？ | 驱动声明 `4x8BitPackedSigned/UnsignedAccelerated = true`，按规范语义即"可用少量指令完成"，实操上应落到 Xe 的 dp4a 指令。**但这是驱动行为，不是规范保证**，必须用 E1 微基准验证 IPC。 |
| 额外开销 | Q4_K 需要先解包 nibble：`lo = w & 0x0F0F0F0F`、`hi = (w>>4) & 0x0F0F0F0F`，每条 16 B 权重对应 2 条 AND + 1 条 shift + 2 条 dp4a。CUDA 版同样如此，**这部分不免费也不额外变差**。 |
| 更快的路 | Xe2 的 XMX 支持 **cooperative matrix int8（M=8 N=16 K=32）**，llama.cpp 在 B580/Windows 上已探测到（[#30058](https://github.com/ggml-org/llama.cpp/issues/30058)）。prefill 的 MMQ 可改走 `VK_KHR_cooperative_matrix`，实测 int8 GEMM 41 TOPS。**但在 140T iGPU 上是否暴露该扩展：未查证**（B580 上是 dGPU）。 |

### 3.6 现有 CUDA 内层到 GLSL 的逐行对照

```cuda
// 现状（q4k_gemv.cu）：一次读 16 B 权重，低/高 nibble 各做一次 dp4a
dlo = __dp4a(wv.x & 0x0F0F0F0F, al.x, dlo);
dhi = __dp4a(wv.x >> 4,        ah.x, dhi);
```

```glsl
// Vulkan / GLSL
#extension GL_EXT_shader_integer_dot_product : require
uint w = wv.x;
dlo = dotPacked4x8EXT(w & 0x0F0F0F0Fu, al.x);
dhi = dotPacked4x8EXT((w >> 4) & 0x0F0F0F0Fu, ah.x);  // uint 是逻辑右移，必须再掩一次
```

| 注意点 | 说明 |
|---|---|
| 移位语义 | GLSL 的 `>>` 对 `uint` 是逻辑右移，高位补 0；CUDA 侧 `int` 右移是算术右移，**必须补 `& 0x0F0F0F0F` 才等价** |
| 类型 | 扩展要求 `int`/`uint` 成对；混用 signed/unsigned 走的是 MixedSignedness 特性位，仍为 accelerated |
| 累加 | `dotPacked4x8EXT` 返回 int，最终统一转 float 再 `fma` scale，与 CUDA 版一致 |

---
## 4. 性能模型

**解码是纯带宽题**：每生成 1 token 要把全部权重读一遍（GEMV 每权重只参与 1 次 MAC）。

```
tok/s_max = BW_eff / (W_bytes + KV_bytes_per_token + 其它流量)
W_bytes  = 模型量化后字节数
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × ctx × 2 B (fp16)
```

| 模型 | 权重 | 136.5 GB/s 理论 | 100 GB/s 现实带宽 | 现实系数 | **预期区间** |
|---|---|---|---|---|---|
| 1.5B Q4_K | 0.81 GiB = 0.87 GB | 157 tok/s | 115 tok/s | 0.50–0.70（小模型固定开销占比高） | **57–80 tok/s** |
| 7B Q4_K | 3.99 GiB = 4.28 GB | 31.9 tok/s | 23.3 tok/s | 0.55–0.75 | **13–17 tok/s** |

**长上下文修正（7B / Qwen2.5 口径：28 层、4 KV heads、head_dim 128）**：每 token 读 KV = 28×4×128×2×2 = 57 KB。

| ctx | KV 读取量 | 占权重比 | 7B 预期 |
|---|---|---|---|
| 4k | 224 MiB | 5% | 13–17 tok/s |
| 32k | 1.75 GiB | 44% | **9–12 tok/s** |

**现实系数的三个扣减项**：
1. **iGPU 抢带宽**：核显与 CPU 共享 136.5 GB/s。分词、采样、日志、桌面合成（浏览器/视频）都会挤占。实测桌面会话比纯 TTY 慢 **15–20%**（[lhl](https://github.com/lhl/intel-inference)）。
2. **温降**：小新 Pro 14 机身 16.9 mm、双风扇，iGPU TGP 35 W 且与 CPU 共享 package 功耗。同类轻薄本已出现"高功耗墙导致持续性能下滑"的记录（[Notebookcheck: Yoga Pro 7 / Ultra 9 高功耗墙](https://www.notebookcheck.net/Lenovo-s-Yoga-Pro-7-with-Intel-s-Core-Ultra-9-shows-the-problem-with-high-power-limits.1161238.0.html)）。持续 30 分钟后按 **-10~-25%** 预估。
3. **dispatch 开销**：llama.cpp 在 258V 上 1B Q4_K_XL 只有 48.5 tok/s（≈38 GB/s 有效带宽），远低于带宽上限 —— 说明**每 token 几十次 dispatch 的 CPU 提交开销才是小模型的瓶颈**（[lhl](https://github.com/lhl/intel-inference)）。我们的单体引擎若能把每 token dispatch 数压到 <20，1.5B 能显著超过这个基线。

**Prefill（计算受限，另一套账）**：dp4a 峰值 2.3 T/s（9.2 TMAC/s）；7B 每 token 1.38e10 MAC → 100% 时 667 tok/s，按 25–35% MAC 利用率 → **150–250 tok/s**；若走 cooperative matrix int8，上限可到 41 TOPS 量级，但受 32 KB smem 与 iGPU 频率限制，先按同量级估。

---
## 5. 移植方案

### 5.1 总体形态
| 决策 | 选择 | 理由 |
|---|---|---|
| 内存 | **UMA device-local buffer 常驻权重**，CPU 侧不保留副本 | 32 GB 内存要放 4 GB 权重 + KV + 系统，双份会 OOM |
| 权重分块 | 按 ≤ **1 GiB** 切片（如每 8–16 层一块），descriptor 数组索引 | `maxStorageBufferRange = 1.0 GiB`（实测） |
| 参数传递 | **push constant（≤256 B）** 传 offset/指针/标量，descriptor 一次性写好不再改 | 避免每 token `vkUpdateDescriptorSets` |
| 提交 | **每 token 一个 command buffer**，一次 submit，一次 fence | WDDM 下每次 submit ~几十 µs，是 1.5B 的主要对手 |
| dispatch 粒度 | 一个算子一个 pipeline、一个 dispatch；**能融合的必须融合**（RMSNorm+quant、SwiGLU+quant、residual+quant） | 减少 barrier 与调度开销 |
| 同步 | 只在"跨 workgroup 读写同一 buffer"处插 barrier；优先 `subgroupBarrier`/`memoryBarrierBuffer` | Intel 全局 barrier 昂贵 |
| 权重布局 | 保持 GGUF Q4_K 原样（144 B/256 权重），**不重排** | 省一次 4 GB 重写；解包成本与 CUDA 版相同 |

### 5.2 十个算子逐个映射
| # | 算子 | CUDA 现状 | Vulkan 方案 | 处置 |
|---|---|---|---|---|
| 1 | `quant` fp32→int8 | 1 warp/行，每 32 元素一组 scale+sum | 同构；`subgroupMax`+`subgroupAdd` 替代 shuffle | ✅ **直接搬** |
| 2 | Q4_K GEMV | 1 warp = 1 行，lane 固定 8 权重 + 三级流水 | 1 subgroup(32) = 1 行；`dotPacked4x8EXT`；预取用 `int4`/`uvec4` | ✅ **直接搬**（主力路径，先做这个） |
| 3 | MMQ 瓦片 | 64 行 × 32 token，smem 17.7 KB，1 load / 5 dp4a | 同构可放下，但 **32 KB 上限**限制双缓冲 | ⚠️ **重写**（缩瓦片或改 coopmat int8） |
| 4 | RMSNorm | block 归约 + smem | workgroup 1024 上限内；两级归约（subgroup→smem） | ✅ 直接搬 |
| 5 | RoPE | 逐元素，查 cos/sin 表 | 同构；表放 storage buffer | ✅ 直接搬 |
| 6 | attention decode（split-K） | KV 切 32 段，`part[head][split]` + combine | 同构；partial buffer 用 device-local；combine 单独 dispatch | ✅ 骨架搬，combine 重写 |
| 7 | attention prefill（flash） | `__shfl` 归约 + 片段式 softmax | 无 `ldmatrix`/async copy，寄存器压力大 | ⚠️ **重写**（先做正确性，再谈性能） |
| 8 | embed（Q4_K 取行） | 1 block/token | 同构 | ✅ 直接搬 |
| 9 | residual / SwiGLU | 逐元素 | 并入前一算子的 dispatch（省一次往返） | ✅ 直接搬 + 融合 |
| 10 | sample greedy/top-k | GPU 归约 + 回读 | **留在 CPU**：一次 4 KB 回读 + 同步，比 GPU 归约划算 | ❌ **不搬** |

### 5.3 compute shader 组织细则

1. **一个算子一个 `.comp`**，模型维度（K、rows、n_head、head_dim）全部走 `layout(constant_id)` specialization constant，编译期定死 —— 与现有 `template<int K>` 策略一一对应。
2. **descriptor**：一个 descriptor set，绑定 N 个权重块（≤1 GiB）+ 激活/KV/输出；**整个会话只写一次**。
3. **push constant** 装：权重块索引/偏移、token 位置、行区间、scale 指针偏移。256 B 够用。
4. **不要用 SSBO 拷贝路径喂激活**：实测 H2D/D2H 只有 45–54 GB/s，直接让 CPU 写 host-visible 的激活 buffer，GPU 侧 `vkCmdPipelineBarrier` 前 flush。
5. **prefill 分块**：WDDM 有 2 s TDR 上限，单次 dispatch 必须远小于 2 s —— 7B 长 prompt 必须按 tile 切分并分批 submit。

### 5.4 CUDA → Vulkan 概念映射（改写时的对照表）
| CUDA | Vulkan compute | 备注 |
|---|---|---|
| `warp`（32 线程） | `subgroup`（32） | 用 `VK_EXT_subgroup_size_control` 显式要求 32 |
| `__shfl_down_sync` | `subgroupShuffleDown` | 归约优先 `subgroupAdd`/`subgroupMax` |
| `__syncthreads()` | `barrier()` | 只同步 workgroup 内 |
| `__shared__` | `shared` 变量 | **32 KB 上限（Windows）** |
| `blockIdx/gridDim` | `gl_WorkGroupID/gl_NumWorkGroups` | 每维 ≤65536，大 grid 要自己切 |
| `__launch_bounds__` | `layout(local_size_x=N)` | ≤1024 |
| `__restrict__` | `restrict` | 帮助驱动做 LICM |
| kernel 参数 | push constant / UBO | **≤256 B**，装偏移与标量，指针用 BDA |
| `atomicAdd(float)` | 需 `VK_EXT_shader_atomic_float` | **140T 是否支持未查证** → split-K combine 改两段式，不用 float 原子 |
| `cudaMalloc` 大块 | `vkAllocateMemory` ≤3.99 GiB、descriptor ≤1 GiB | 分块是硬要求 |

---
## 6. 风险与未知

| 风险 | 已知证据 | 缓解 |
|---|---|---|
| **Intel 驱动 int8 点积正确性/稳定性** | ① Arc A770 上 int8 dot 触发 `ErrorDeviceLost`（[#30692](https://github.com/ggml-org/llama.cpp/issues/30692)）；② 驱动区间 **[32.0.101.8509, 32.0.101.8860)** 错编译，社区直接按版本禁用 kernel（[PR #27049](https://github.com/ggml-org/llama.cpp/pull/27049)）；③ Xe2 coopmat int8 瓦片尺寸不匹配导致列丢失（[#30058](https://github.com/ggml-org/llama.cpp/issues/30058)） | 固定驱动版本并写进 README；核心算子保留 **fp16 回退路径**；每次构建跑逐层数值对拍（现成 `tests/t_gemv` 思路） |
| **持续满载降频** | 35 W iGPU + 轻薄机身；同类机型有高功耗墙记录（[Notebookcheck](https://www.notebookcheck.net/Lenovo-s-Yoga-Pro-7-with-Intel-s-Core-Ultra-9-shows-the-problem-with-high-power-limits.1161238.0.html)） | 基准必须跑 **30 分钟**并报"第 1 分钟 / 第 30 分钟"两个数；必要时锁 PL |
| **显存分配失败** | 单次分配 ≤3.99 GiB、单 buffer ≤1.0 GiB（实测）；系统内存被其它进程占用 | 分块 + 启动时探测 heap 大小 + 失败时降级（部分层跑 CPU） |
| **WDDM TDR（2 s）** | Windows 默认行为 | 单 dispatch 限时；长 prefill 切片 |
| **驱动自动更新** | Windows Update 会静默换驱动版本，可能落进坏区间 | 组策略/注册表锁版本；启动时打印驱动版本 |
| **coopmat 在 iGPU 上是否可用** | 只在 B580(dGPU) 上有公开证据 | **未查证**，E1 里顺带探测 |
| **实际内存型号** | 有资料写 DDR5-5600，与题设 LPDDR5x-8533 冲突 | 上机用 `wmic memorychip` / CPU-Z 核对，差 1.5× 带宽 |
| **共享显存"一半物理内存"规则** | 未能取得官方文档 | **未查证**，以 `vkGetPhysicalDeviceMemoryProperties` 实测为准 |

---
## 7. 一周验证计划（三个最小实验）

### E1｜能力探针 + dp4a 微基准（0.5 天）

* **做什么**：写一个 `vulkaninfo` 级别的探针，打印 deviceName / driverVersion / apiVersion / `maxComputeSharedMemorySize` / `maxMemoryAllocationSize` / `maxStorageBufferRange` / subgroup 范围 / `shaderInt8` / 四个 `integerDotProduct*Accelerated` / `VK_KHR_cooperative_matrix`；再写两个 kernel：A = 纯 `dotPacked4x8EXT` 循环，B = 手工 `(a&0xF)*...` 展开 4 次乘加，同数据同循环数比时间。
* **判定**：① 探针数值与本文件 §1.2/§3 一致（不一致以实测为准，回来改文档）；② **A 比 B 快 ≥1.8×** → 确认一对一 dp4a 指令成立；若只快 <1.2× → 被拆成多条，Q4_K 内层要重新设计；③ 探针能报出 coopmat 是否可用（决定 MMQ 走 dp4a 还是 XMX）。

### E2｜带宽实测 + 单层 Q4_K GEMV（1.5 天）

* **做什么**：① 用 4 GB buffer 做 `y[i]=x[i]+1` 的 stream 测试，量有效 GB/s；② 实现 Q4_K GEMV 单层（K=4096, R=4096, M=1），权重 2.25 MB 级，跑 10 万次取均值。
* **判定**：① 有效带宽 ≥ **90 GB/s**（否则说明 UMA 分配或访问模式有问题，先修再谈 kernel）；② 单层 GEMV 达到 **≥70% 实测带宽**（即 ≥63 GB/s，等价 7B 解码 ≥14 tok/s）；③ 与 CPU AVX-VNNI 版本逐元素对拍，误差在 fp32 舍入内。

### E3｜端到端 decode + 30 分钟持续负载（2 天）

* **做什么**：接上 1.5B 与 7B 全链路，测 pp512 / tg128，插电与电池各一轮；连续跑 30 分钟记录 tok/s 与 GPU 频率曲线。
* **判定**：① 7B 落在 **13–17 tok/s**、1.5B 落在 **57–80 tok/s**（模型偏差 >25% 即需修模型参数，而不是"再优化"）；② 第 30 分钟 ≥ 第 1 分钟的 **80%**（否则温降系数要写进产品预期）；③ 全程无 `VK_ERROR_DEVICE_LOST`，日志打印驱动版本备查。

> 三个实验的优先级：**E2 的 ①（带宽）> E1 的 ②（dp4a 是否真加速）> E3**。带宽不达标时后面全部数字作废。

### 7.4 排期
| 天 | 任务 | 产出 |
|---|---|---|
| D1 | 驱动锁版本 + 探针（E1 前半） | capability 报告（填回本文 §1.2/§3） |
| D2 | dp4a 微基准（E1 后半）+ stream 带宽（E2 ①） | 两个数：dp4a 加速比、有效 GB/s |
| D3–D4 | Q4_K GEMV 单层 + CPU 对拍（E2 ②③） | 单层吞吐、误差表 |
| D5–D6 | 端到端 1.5B / 7B（E3 ①②） | tok/s 表（pp512 / tg128） |
| D7 | 30 min 持续负载（E3 ③）+ 回填本文数字 | 定稿：把"引用值"换成"本机值" |

---
## 参考

1. Intel ARK — Core Ultra 7 255H 规格：https://www.intel.com/content/www/us/en/products/sku/241751/intel-core-ultra-7-processor-255h-24m-cache-up-to-5-10-ghz/specifications.html （**本机网络不可达**，仅作官方索引）
2. NanoReview — Core Ultra 7 255H / Arc 140T（1024 SU、2250 MHz、4.6 TFLOPS、35 W）：https://nanoreview.net/en/cpu/intel-core-ultra-7-255h
3. Vulkan Hardware Database #44273（Win11 / 驱动 101.8250 / Vulkan 1.4.325 / 全部 limits 与 features）：http://vulkan.gpuinfo.org/displayreport.php?id=44273
4. Vulkan Hardware Database #31555（驱动 101.5593，对照）：http://vulkan.gpuinfo.org/displayreport.php?id=31555
5. `VK_KHR_shader_integer_dot_product` 参考页（Vulkan 1.3 核心）：https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_integer_dot_product.html
6. `VkPhysicalDeviceShaderIntegerDotProductProperties`（accelerated 语义）：https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderIntegerDotProductProperties.html
7. llama.cpp Vulkan Q4_K/MMQ 着色器（`dotPacked4x8EXT` 用法）：https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl
8. llama.cpp #30692 — int8 dot 在 A770 上 device lost：https://github.com/ggml-org/llama.cpp/issues/30692
9. llama.cpp #27049 — 按 Intel 驱动版本区间禁用 kernel：https://github.com/ggml-org/llama.cpp/pull/27049
10. llama.cpp #30058 — Xe2 coopmat int8 瓦片尺寸 bug（含 B580 驱动上报 coopmat_int 8×16×32）：https://github.com/ggml-org/llama.cpp/issues/30058
11. llama.cpp #18178 — Intel Xe2/Xe3 Vulkan warptile 调优：https://github.com/ggml-org/llama.cpp/pull/18178
12. lhl/intel-inference — Arc 140V 实测带宽（93–107 GB/s）、int8 GEMM 41 TOPS、llama.cpp Vulkan 吞吐：https://github.com/lhl/intel-inference
13. Notebookcheck — Arc 140T vs Radeon 890M（Arrow Lake 首测）：https://www.notebookcheck.net/Arc-140T-vs-Radeon-890M-benchmarks-Both-a-win-and-a-lose-for-Intel.979421.0.html
14. Notebookcheck — 轻薄本高功耗墙问题（Yoga Pro 7 / Ultra 9）：https://www.notebookcheck.net/Lenovo-s-Yoga-Pro-7-with-Intel-s-Core-Ultra-9-shows-the-problem-with-high-power-limits.1161238.0.html
15. canitrun — Arc 140V 带宽 137 GB/s 与第三方 token 估算：https://canitrun.dev/gpus/arc-140v-16/
16. ZOL — 小新 Pro 14 2025 参数（内存类型待核）：https://nb.zol.com.cn/977/9779201_all.html

> 本文所有「实测值」均来自公开的 Vulkan 硬件数据库与第三方实测仓库，**未在本机（无 Arc 140T）验证**；第 7 节的三个实验就是用来把"引用数字"换成"本机数字"的。
