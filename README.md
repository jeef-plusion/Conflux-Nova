# 🚀 Conflux Nova V2.0.0-Core
**⚠️ 注意：之前的所有旧版本架构已全部废除，V2.0.0 从底层进行了纯血单体架构重构。**

一个只跑在 GTX 1080 Ti（sm_61）上的单体推理引擎：**一个二进制 = 一个模型**，没有 graph 层、没有 device 层、没有运行期 dispatch。
为 **Agent 编排** 与 **多角色扮演** 而生：每会话独立 KV、独立流、独立输出缓冲，四路并发共享一次权重读取。

## 🌟 性能实测 (1080Ti, Qwen2.5-7B Q4_K_M, ctx=2048)
- Prefill: 536+ tok/s
- Decode: 53+ tok/s (满血可达 55+ tok/s)
- 显存占用: ~7.1 GB

## 🤖 支持模型清单

| 构建宏 | 模型 | 尺寸 | 权重 | Chat 模板 | 状态 |
|---|---|---|---|---|---|
| `NOVA_MODEL_QWEN25_7B` | Qwen2.5-7B-Instruct | 7B | 3.99 GiB | ChatML | ✅ 实测通过 |
| `NOVA_MODEL_R1_QWEN7B` | DeepSeek-R1-Distill-Qwen-7B | 7B | 3.99 GiB | ChatML | ✅ 实测通过 |
| `NOVA_MODEL_QWEN25_1P5B` | Qwen2.5-1.5B-Instruct | 1.5B | 0.81 GiB | ChatML | ✅ 实测通过 |
| `NOVA_MODEL_R1_QWEN1P5B` | DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.81 GiB | ChatML | ✅ 实测通过 |
| `NOVA_MODEL_R1_QWEN1P5B_ABL` | DeepSeek-R1-Distill-Qwen-1.5B-Abliterated | 1.5B | 0.93 GiB | DeepSeek | ✅ 实测通过 |
| `NOVA_MODEL_R1_LLAMA8B` | DeepSeek-R1-Distill-Llama-8B | 8B | ~4.6 GiB | ChatML | ⚠️ 模具就绪，未实测 |

同时支持 **Qwen2.5 架构下的所有微调变种**：只要维度一致，theta / tied / 模板不同都能通过新增一条模具 + 一张模型卡接入，内核零改动。

### 变种适配矩阵

| 变种差异 | 处理方式 | 实例 |
|---|---|---|
| `rope_theta` 不同（1e4 vs 1e6） | 模具参数 | Abliterated 版 |
| `tie_word_embeddings` 不同 | 模具参数 + 加载器按张量表判定 | Abliterated 版 |
| Chat 模板不同（ChatML / DeepSeek） | `ModelCard.asst` 字段 | 两种模板并行支持 |

## 🎭 为什么适合 Agent 与角色扮演

- **多会话状态隔离**：每路 Session 独占 KV 切片（112 MB）、独立 `cudaStream_t`、独立输出缓冲，四路同跑不串台。
- **增量 prefill**：第二轮只计算新增 token（实测 16 tok / 62 ms），历史 KV 全部复用。
- **批量 decode**：`gemv4` 让 4 路共享一次 Q4_K 权重读取，单步同时产出 4 个 token。
- **批量 attention**：`attn_batch4_kernel` 用一次 launch 覆盖 4 路（每 warp 一个 session-head），消灭 112 次启动。
- **两种对话协议**：ChatML（Qwen 系）与 DeepSeek 模板（`<｜User｜>` / `<｜Assistant｜>`）可在模型卡里切换。
- **流式输出**：逐 token 触发输出链，UTF-8 边界缓冲保证中文不裂。
- **可控采样**：top-k / top-p / temperature（设备端 argmax + 主机端采样）。

### 并发实测（7B，四路角色辩论）

| 场景 | 单路 | 聚合 |
|---|---|---|
| 4 路批量 decode，KV≈100 | 25.9 tok/s | **103 tok/s** |
| 4 路批量 decode，KV≈220 | 15.4 tok/s | **62 tok/s**（P2 归一） |
| 3 路批量 decode，KV≈300 | 26.7 tok/s | **80 tok/s** |
| 1.5B 四路批量，KV≈290 | 75.9 tok/s | **303 tok/s** |

> 这张卡会掉 P5（734/810 MHz），同一份代码能测出 18 ms 和 143 ms 两种结果。
> 所有数字都在同一会话背靠背、`nvidia-smi` 记录到 SM 1690~1847 MHz / MEM 5508 MHz 时采集；P5 数据已按频率归一。

## 🏗 架构

```
INPUT_CHAT → INPUT_TOKENIZR → INPUT_START →（计算）→ OUT_TKZ → OUT_chat → OUT_OUT
```

六段全局函数，每段文件即接口，各自可被独立替换；每段实现完直接唤醒下一段。

| 目录 | 职责 |
|---|---|
| `code/` | 前端六段 + 总装（`main.h` 全部参数在此） |
| `code/kernel/1080ti/` | 计算侧入口与内核聚合 |
| `code/map/qwen2.5/` | 模型卡（模板/系统提示）+ 字节级 BPE |
| `src/kernel/` | GEMV / gemv4 / MMQ / attention / norm / rope（数值单一真源） |
| `src/loader/` | GGUF 解析 + 特征签名校验 |
| `tools/` | Q6_K/Q8_0 → Q4_K 离线转换、张量比对 |

## 🔧 编译与运行

```
build_core.bat NOVA_MODEL_QWEN25_7B
build/nova_core.exe build/q4k.gguf
```

等价的 nvcc 命令：

```bash
nvcc -std=c++17 -O3 -arch=sm_61 -Xcompiler /utf-8 -DNOVA_MODEL_QWEN25_7B -Iinclude -Icode ^
     code\kernel\1080ti\gemv_q4k.cu code\kernel\1080ti\attn.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp code\main.cpp ^
     -o build\nova_core.exe
```

## 🙏 致谢与开源声明
本项目在底层研发过程中，深度参考并借鉴了 **[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)** 的优秀设计。特别是针对 Pascal 架构的显存带宽压榨、Q4_K 的 Nibble 解包逻辑、以及 Split-K Attention 的实现，深受其启发。本项目遵循开源规范，在此向 llama.cpp 的开发者及所有极客致以最崇高的敬意。

> ## ⛔ 版本支持策略（2026-10 起生效）
> **V2.0.0-Core 之前的任何版本、PR、Issue、补丁、分支，一律不再支持。**
>
> - 旧架构（Neo 五层分层：`code/graph`、`code/device`、`code/chat`、`code/user`、`code/kernel` 包装层）已整体废弃，并从 `main` 移除。
> - v1.x 系列、`v1.2.0-neo` 及更早的 tag，仅作为历史归档保留，不再接受修复、不再保证可编译。
> - 基于旧接口的 PR 会被直接关闭，不进入评审。
> - 唯一受支持的接口、行为与性能基线，以本仓库 `main` 分支的 V2.0.0-Core 为准。

## 已知限制

- stock Q4_K_M 的 `attn_v` / `ffn_down` 是 Q6_K，本模具只吃纯 Q4_K，需先跑 `tools/requant_q4k.cpp` 离线转换。
- 分词器是简化版字节级 BPE（无完整 Unicode 正则预切分），中英混排基本准确。
- 单次 `cudaMalloc` 上限实测约 6 GB（WDDM），14B 档需拆分配。
