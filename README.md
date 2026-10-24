# 🚀 Conflux Nova V2.0.0-Core
**⚠️ 注意：之前的所有旧版本架构已全部废除，V2.0.0 从底层进行了纯血单体架构重构。**

## 🌟 性能实测 (1080Ti, Qwen2.5-7B Q4_K_M, ctx=2048)
- Prefill: 536+ tok/s
- Decode: 53+ tok/s (满血可达 55+ tok/s)
- 显存占用: ~7.1 GB

## 🙏 致谢与开源声明
本项目在底层研发过程中，深度参考并借鉴了 **[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)** 的优秀设计。特别是针对 Pascal 架构的显存带宽压榨、Q4_K 的 Nibble 解包逻辑、以及 Split-K Attention 的实现，深受其启发。本项目遵循开源规范，在此向 llama.cpp 的开发者及所有极客致以最崇高的敬意。

> ## ⛔ 版本支持策略（2026-10 起生效）
> **V2.0.0-Core 之前的任何版本、PR、Issue、补丁、分支，一律不再支持。**
>
> - 旧架构（Neo 五层分层：`code/graph`、`code/device`、`code/chat`、`code/user`、`code/kernel` 包装层）已整体废弃，并从 `main` 移除。
> - v1.x 系列、`v1.2.0-neo` 及更早的 tag，仅作为历史归档保留，不再接受修复、不再保证可编译。
> - 基于旧接口的 PR 会被直接关闭，不进入评审。
> - 唯一受支持的接口、行为与性能基线，以本仓库 `main` 分支的 V2.0.0-Core 为准。
>
> 需要旧版本请用 tag 自行检出；我们不提供迁移补丁，只提供这一份 V2.0.0-Core。


---

## 这是什么

一个只跑在 GTX 1080 Ti（sm_61）上的单体推理引擎：**一个二进制 = 一个模型**，没有 graph 层、没有 device 层、没有运行期 dispatch。

| 能力 | 状态 |
|---|---|
| Qwen2.5-7B / 1.5B、R1-Distill-Qwen-7B/1.5B、R1-Distill-Llama-8B | 编译期模具 |
| Q4_K 权重 × Q8 激活（dp4a） | 已落地 |
| MMQ 瓦片（64 行 × 32 token，smem 复用） | 已落地 |
| Split-K Attention（4 warp / block，block 内 combine） | 已落地 |
| 字节级 BPE + ChatML 前端 | 已落地 |
| 权重一次 cudaMalloc + pinned DMA 上传 | 已落地 |

## 目录

    code/
    ├── USER_INPUT_TXT.h     文本 → ChatML
    ├── INPUT_INPUT.h        模板串 → Token
    ├── OUT_TKZ.h            Token → 文本
    ├── OUT_chat.h           输出清洗
    ├── OUT_OUT.h            最终输出 + 计时
    ├── kernel/1080ti/       gemv_q4k.cu / attn.cu / kernel_api.h
    ├── map/qwen2.5/         model_map.h / tokenizer.h
    ├── main.h               配置 + 控制台入口
    └── main.cu
    src/kernel/              GEMV / MMQ / norm / rope / attn（数值单一真源）
    src/loader/              GGUF 解析 + 特征签名校验
    src/model.cpp            权重上传 + 逐层 forward

## 编译

    nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
         -DNOVA_MODEL_QWEN25_7B -Iinclude -Icode ^
         code\kernel\1080ti\gemv_q4k.cu code\kernel\1080ti\attn.cu ^
         src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp code\main.cu ^
         -o build\nova_core.exe

## 运行

    build\nova_core.exe build\q4k.gguf

## 实测记录（同一会话、满血 P2 档）

| 项 | 数值 |
|---|---|
| prefill 2048 tok | 536.3 tok/s |
| decode 128 tok | 55.24 tok/s（18.1 ms/tok） |
| gemv 段带宽 | 298 GB/s |
| 显存峰值 | 7095 MB |
| 显存池 | 4.924 / 5.011 GiB |

> 这张卡会掉 P5 档（746/810 MHz），同一份代码能测出 25 ms 和 98 ms 两种结果。所有数字都在同一会话背靠背、nvidia-smi 记录到 SM 1690~1847 MHz / MEM 5508 MHz 时采集。

## 模型适配

模具是编译期常量。换模型 = 换 -DNOVA_MODEL_* 重新编译；加载时用 GGUF metadata 做特征签名校验（arch / layers / d_model / heads / head_dim / ffn / vocab / theta / eps / tie），签名一致但名字不在白名单时黄字 [WARN] 放行；维度或量化路径不符直接 abort。

## 已知限制

- stock Q4_K_M 的 attn_v / ffn_down 是 Q6_K，本模具只吃纯 Q4_K，需先跑 tools/requant_q4k.cpp 离线转换。
- 分词器是简化版字节级 BPE（无完整 Unicode 正则预切分），中英混排基本准确。
