# 🚀 Conflux Nova V2.0.0-Core

**当前官方编译版本：V2.1.0** ｜ 架构版本：V2.0.0-Core

一个只跑在 GTX 1080 Ti（sm_61）上的单体推理引擎：**一个二进制 = 一个模型**，没有 graph 层、没有 device 层、没有运行期 dispatch。
为 **Agent 编排** 与 **多角色扮演** 而生：每会话独立 KV、独立流、独立输出缓冲，四路并发共享一次权重读取。

## 🎬 它跑起来是什么样

四个角色，一块 1080 Ti，各自持有自己的 KV 与系统提示词，同一条命令：

```text
########## ROUND 1 ##########
[A 咸粽党]  咸粽才是正宗！蛋黄五花肉才是灵魂！
[B 甜粽党]  甜粽才是王道，豆沙蜜枣才是经典！
[C 辣椒邪教] 我都反对，应该咸甜混合加辣椒芝士榴莲！拱火才有趣！
[D 不爱吃]  都不想吃，粽子对我来说就是碳水，无所谓咸甜。

########## ROUND 2 ##########
[A 咸粽党]  咸粽才是唯一真理！蛋黄五花肉才是王道！
[B 甜粽党]  甜粽才是正宗！豆沙蜜枣馅儿，Q弹糯米，甜而不腻，才是我的心头好！
[C 辣椒邪教] 俩都无聊！加辣椒芝士榴莲才刺激！拱火才开心！
[D 不爱吃]  咸甜皆非吾所好，唯求别物。

########## ROUND 3 ##########
[A 咸粽党]  咸粽才是王道！蛋黄五花肉才正宗！
[B 甜粽党]  甜粽才是唯一真理！豆沙蜜枣馅，Q弹糯米，甜甜蜜蜜才是王道！
[C 辣椒邪教] 俩都假的！真香是辣椒芝士榴莲馅！拱火才真实！
[D 不爱吃]  非吾所好。
```

每一轮，四路都会拿到其他三路上一轮的发言，各自带着自己的历史和人格继续吵。
## 🌟 性能实测 (1080Ti, Qwen2.5-7B Q4_K_M, ctx=2048)
- Prefill: 536+ tok/s
- Decode: 53+ tok/s (满血可达 55+ tok/s)
- 显存占用: ~7.1 GB

---

## ⛔ 版本支持策略（2026-10-23 起生效）

**V2.0.0-Core 之前的任何版本、PR、Issue、补丁、分支，一律不再支持。**

- 旧架构（Neo 五层分层：`code/graph`、`code/device`、`code/chat`、`code/user`、`code/kernel` 包装层）已整体废弃，并从 `main` 移除。
- v1.x 系列、`v1.2.0-neo` 及更早的 tag，仅作为历史归档保留，不再接受修复、不再保证可编译。
- 基于旧接口的 PR 会被直接关闭，不进入评审。
- 唯一受支持的接口、行为与性能基线，以本仓库 `main` 分支的 V2.0.0-Core 为准。

需要旧版本请用 tag 自行检出；我们不提供迁移补丁，只提供这一份 V2.0.0-Core。

---

## 🤖 支持模型清单

| 构建宏 | 模型 | 尺寸 | 权重 | Chat 模板 | 状态 |
|---|---|---|---|---|---|
| `NOVA_MODEL_QWEN25_7B` | Qwen2.5-7B-Instruct | 7B | 3.99 GiB | ChatML | ✅ 已实测 |
| `NOVA_MODEL_R1_QWEN7B` | DeepSeek-R1-Distill-Qwen-7B | 7B | 3.99 GiB | ChatML | ✅ 已实测 |
| `NOVA_MODEL_QWEN25_1P5B` | Qwen2.5-1.5B-Instruct | 1.5B | 0.81 GiB | ChatML | ✅ 已实测 |
| `NOVA_MODEL_R1_QWEN1P5B` | DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.81 GiB | ChatML | ✅ 已实测 |
| `NOVA_MODEL_R1_QWEN1P5B_ABL` | DeepSeek-R1-Distill-Qwen-1.5B-Abliterated | 1.5B | 0.93 GiB | DeepSeek | ✅ 已实测 |
| `NOVA_MODEL_R1_LLAMA8B` | DeepSeek-R1-Distill-Llama-8B | 8B | ~4.6 GiB | ChatML | ⚠️ 模具就绪 |

### 变种适配矩阵

| 变种差异 | 处理方式 | 已验证实例 |
|---|---|---|
| `rope_theta` 不同（1e4 vs 1e6） | 模具参数 | Abliterated 版 |
| `tie_word_embeddings` 不同 | 模具参数 + 加载器按张量表判定 | Abliterated 版 |
| Chat 模板不同（ChatML / DeepSeek） | `ModelCard.asst` 字段 | 两种模板并行支持 |

## 🎭 为什么适合 Agent 与角色扮演

- **多会话状态隔离**：每路 Session 独占 KV 切片（112 MB）、独立 `cudaStream_t`、独立输出缓冲，四路同跑不串台。
- **增量 prefill**：第二轮只计算新增 token（实测 16 tok / 62 ms），历史 KV 全部复用。
- **批量 decode**：`gemv4` 让 4 路共享一次 Q4_K 权重读取，单步同时产出 4 个 token。
- **两种对话协议**：ChatML（Qwen 系）与 DeepSeek 模板（`<｜User｜>` / `<｜Assistant｜>`）可在模型卡里切换。
- **流式输出**：逐 token 触发输出链，UTF-8 边界缓冲保证中文不裂。

### 并发实测（7B，四路角色辩论）

| 场景 | 单路 | 聚合 |
|---|---|---|
| 4 路批量 decode，KV≈100 | 25.9 tok/s | **103 tok/s** |
| 4 路批量 decode，KV≈220 | 15.4 tok/s | **62 tok/s**（P2 归一） |
| 3 路批量 decode，KV≈300 | 26.7 tok/s | **80 tok/s** |
| 1.5B 四路批量，KV≈290 | 75.9 tok/s | **303 tok/s** |

> 这张卡会掉 P5（734/810 MHz），同一份代码能测出 18 ms 和 143 ms 两种结果。以上数字均在 `nvidia-smi` 记录到 SM 1690~1847 MHz / MEM 5508 MHz 时采集，P5 数据已按频率归一。

---

## 技术特点

### 1. 模块化设计

在我们的架构中，有一批文件的命名遵循「大写 + 下划线」的格式——它们本质上就是**模块**。

传统架构把词表和聊天模板嵌在内部，想单独改一处十分困难。我们直接把它们拆成独立文件；使用时，模块的入口是一个函数，参数由文件自己构造。

这给了开发者极大的自由，同时明确了分工：对不同领域有不同擅长的人，完全可以提供自己的逻辑，在编译时由他人复用、自行编辑参数。

```
INPUT_CHAT → INPUT_TOKENIZR → INPUT_START →（计算）→ OUT_TKZ → OUT_chat → OUT_OUT
```

每段文件即接口，实现完直接唤醒下一段；换一段实现，其余五段不知道也不关心。

```
code/
├── INPUT_CHAT.h         收用户输入 → 套聊天模板
├── INPUT_TOKENIZR.h     文本 → Token
├── INPUT_START.h        计算侧针脚（只吃 token id）
├── OUT_TKZ.h            Token → 文本
├── OUT_chat.h           输出清洗
├── OUT_OUT.h            最终输出 + 计时
├── session.h            会话状态（KV 切片 / 历史 / 输出缓冲）
├── kernel/1080ti/       gemv_q4k.cu / attn.cu / kernel_api.h
├── map/qwen2.5/         model_map.h / tokenizer.h
├── main.h               初始化 + 主循环
└── main.cpp
src/kernel/              GEMV / MMQ / norm / rope / attn（数值单一真源）
src/loader/              GGUF 解析 + 特征签名校验
src/model.cpp            权重上传 + 逐层 forward
```

### 2. 计算物理化

传统架构总想干一件事：让什么硬件都能算什么。但这里必须意识到一个物理事实——1080 Ti 只有 11 GB 显存，它能跑的模型上限在哪？超过 11 GB 就溢出到内存，此时计算极低效，体验完全谈不上。

我们提出一个描述方式：把硬件的计算速度抽象成函数 `v(d)`，数据是一个复杂矢量 `d`，计算量为 `s`，则耗时 `t = s / v(d)`。

`d` 非常复杂，要结合数据量、数据类型、数据格式。比如给一个 64 位 CPU 喂「每块 38 字节」或「每块 257 字节」的数据，它受不受得了？

所以优化必须结合实际：1080 Ti 怎么可能去跑 27B 模型？用 Q2 这种级别的量化，跑出来的东西能用吗？那还不如直接跑一个 1.5B 的低量化。

在我们的架构里，适配一个模型不再看它的名字，而是看它的实际实力。

### 3. 计算的绝对黑盒性（仅针对前端与应用层）

传统架构里做一个 AI 应用极其复杂：要调这个、要调那个，还不清楚它到底怎么跑。而在这里，你可以在完全不清楚计算细节的情况下直接跑起来。

我们把「接入计算」的成本压缩成了一个文件里的 `OUT_OUT`。

一个编译版本里可以内嵌特别多硬件后端，而这些后端你可以直接从硬件开发者和内核开发者手里拿到他们的库，然后调用我们的全局函数 `OUT_OUT`，就可以把你的东西直接提交给他们（前提当然是有相同的初始化条件）。

在保证了计算的绝对黑盒性之后，上层想怎么玩，随便怎么玩。

### 4. KV cache 哈希编码与对话块管理（设计）

我们给 AI 的每一次输出都编一个 KV cache 块。整体逻辑是：

- **前缀** = 该对话第一次输入量化之后的 ID；
- **后缀** = 该次对话 KV cache 增量的量化 ID。

这样划分的好处是：每一句话是一个基本单位，AI 输出的每一次结果也是一块。人说的话一般比 AI 少，记忆的绝大部分由 AI 的输出占据，而「字」就是最基本的分割单位。

这样划分之后，计算时可以由用户自己决定使用哪些记忆块，从而实现对记忆的高度修改——这在角色扮演里是一件利器。

---

## 发行版声明

本人作为作者拥有架构的更新和迭代能力，但本人发布的任何编译版本不一定代表整体的技术方向。

我不会过度干预各位在各种地方的技术实现。我们只遵循定下的一部分必要的契约和协议，而不是限制接口。

同时欢迎各位向我提交 PR，官方仓库会尽量跟上大家的开发节奏，并遴选出最适合、最广泛、最实用的代码库，供各位参考。

作者个人实力有限。发布的版本如果有问题、有 bug，欢迎提 PR，我们会认真听取意见。

---

## 📦 官方编译版本 V2.1.0 使用说明

> **版本号只标记“第几次官方发布”，不标记架构。**
> 架构始终是 Core：单体、编译期模具、六段模块契约。2.1.0 是我们把这套架构实现出来、跑出效果的第 1 个 2.x 官方编译版本。

### 官方编译版 vs 贡献者编译版

| | 官方编译版 | 贡献者编译版 |
|---|---|---|
| 版本号 | 2.1.0（由官方递增） | 自定 |
| 谁维护 | 官方仓库 | 作者本人 |
| 改动范围 | 架构契约 + 模块实现 | **只改模块**，不动契约 |
| 向下游承诺 | 受支持的接口与行为基线 | 无（自负） |
| 如何进入官方 | — | **提交模块 PR** |

我们接受的是**模块**，不是新架构：六段里的任意一段（`INPUT_CHAT` / `INPUT_TOKENIZR` / `INPUT_START` / `OUT_TKZ` / `OUT_chat` / `OUT_OUT`），只要保持“处理完直接唤醒下一段”的契约，官方仓库会挑选最广泛、最实用的实现合并进来。

### 这个版本里有什么

| 能力 | 说明 |
|---|---|
| 多会话 | `Session` + `SessionManager`，KV 按会话切片，最多 4 路并行 |
| KV 复用 | 第二轮只 prefill 新增 token（实测 16 tok / 62 ms），历史全部复用 |
| 批量 decode | `gemv4`，4 路共享一次 Q4_K 权重读取 |
| 双对话协议 | ChatML（Qwen 系）/ DeepSeek（`<｜User｜>`） |
| 可控采样 | top-k / top-p / temperature |

### 编译

```
build_core.bat NOVA_MODEL_QWEN25_7B
```

括号里的宏换成支持清单里任意一个即可（`NOVA_MODEL_R1_QWEN7B`、`NOVA_MODEL_QWEN25_1P5B` …）。

### 运行

```
build/nova_core.exe
```

`code/main.h` 里内置的样例是**四路角色辩论**：四个 Session、各自一份 system prompt、三轮互相引用对方上一轮的发言。

### 改成你自己的用法

所有可调参数都在 `code/main.h` 顶部的 `cfg::` 里，改完重新编译即可，内核一行不用碰：

```cpp
inline constexpr const char* kModelPath = "build/q4k.gguf";        // 权重的 gguf 路径
inline constexpr const char* kCardName  = "Qwen2.5-7B-Instruct";   // 模型卡（决定模板与 system）
inline constexpr int kCtx    = 2048;   // 单会话上下文上限
inline constexpr int kTurn   = 24;     // 每轮生成多少 token
inline constexpr int kRounds = 3;      // 辩论多少轮
inline constexpr const char* kSys[4]   = { /* 四个角色的设定 */ };
inline constexpr const char* kLabel[4] = { "A 催婚长辈", "B 不婚主义", "C 乐子人", "D 现实主义者" };
```

### 输出形态

```
########## ROUND 1 ##########
[A 催婚长辈] 结婚生子是天伦之乐，不结婚如何传宗接代，老了谁照顾你？
[B 不婚主义] 结婚是个人选择，但婚姻制度本身反人性，是资本操控的工具。
[PERF] round 1: 24 tok x4 in 3437 ms | 7.0 tok/s per session | 27.9 tok/s total
```

---

## 🙏 致谢

本项目在底层研发过程中，深度参考并借鉴了 **[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)** 的优秀设计。特别是针对 Pascal 架构的显存带宽压榨、Q4_K 的 Nibble 解包逻辑、以及 Split-K Attention 的实现，深受其启发。在此向 llama.cpp 的开发者及所有极客致以最崇高的敬意。
