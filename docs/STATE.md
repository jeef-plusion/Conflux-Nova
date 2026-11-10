# 项目现状（2026-11-10 收尾快照）

> 一页纸，给"过几天回来接着干"的人（也包括我自己）。细节在各自文档里，这里只放**结论、数字、下一步、坑**。
> 如果只读一段：**算子层和 1.5B 端到端都跑通了（35.5 tok/s 实测）；DeepSeek-V2-Lite 的注意力/路由算子验完了，
> 卡在缺 Q5_0/Q8_0/Q6_K 的 GEMV 和"权重大于显存"两件事上。**

---

## 1. 现在能做到什么

| 能力 | 状态 | 数字 | 文档 |
|---|---|---|---|
| SYCL 算子数值 | ✅ | 五项 `rel_l2` 1e-7~1e-8，`quant` 抽检 0 不一致 | `PERF_BASELINE.md` §2/§10 |
| GEMV 带宽 | ✅ | 稳态 **36.8~38.8 GB/s** = 顺序读上限 61.38 的 60% | `PERF_BASELINE.md` §8.1 |
| 1.5B 端到端 decode | ✅ | **28.14 ms/token = 35.54 tok/s**（真模型 28 层） | `SYCL_PORT.md` §12 |
| 方案 C（lane 重映射） | ❌ 已否决 | 四次 A/B 全负（-2.7% ~ -9.8%） | `SYCL_PORT.md` §11.6 |
| DeepSeek-V2-Lite | 🟡 算子就绪 | MLA（rope/kv_store/decode）+ MoE 路由全部对拍 PASS | `DEEPSEEK_V2_LITE.md` |
| 门槛 | ✅ 全绿 | 回归门 34 GB/s / 37 tok/s，`sycl_bench` 退出码 0 | `PERF_BASELINE.md` §11 |

## 2. 标签 → 状态

| tag | commit | 是什么 |
|---|---|---|
| `v2.3.0-beta` | `b11e9c3` | 算子层毕业：quant 致命 bug 修完、方案 C 结案、门槛重定 |
| `v2.4.0-alpha` | `c6a7d0f` | 第一次端到端 decode（28 层真模型，35.5 tok/s） |
| HEAD | `d731baa` 之后 | + V2-Lite 的 MLA/MoE 算子、混量化 loader、`quant_dump`/`gguf_dump` 工具 |

## 3. 怎么复现（三条命令，都不需要 GPU 全部）

```bat
:: 1) 算子数值（CPU，只需 MSVC）
build_sycl_host.bat 1.5b 16
build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf

:: 2) V2-Lite 的 MLA/MoE 算子（CPU，不需要模型文件）
build_dsv2_host.bat
build\sycl_dsv2_host_check.exe

:: 3) 端到端 decode（目标机，需 oneAPI + 模型）
build_sycl_e2e.bat 1.5b
build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 24
```

性能测量**必须**按 `PERF_BASELINE.md` §7 的协议（长测量、同轮交替、单进程、`--warm-ms`），
否则会量出 18 GB/s 那种假象。

## 4. 下一步（按依赖顺序，别跳）

1. **Q5_0 / Q8_0 / Q6_K 的 GEMV**：`gemv<>` 现在只认 Q4_K，而 V2-Lite 的 `ffn_down`(Q8_0)、
   `ffn_down_exps`(Q8_0/Q5_0 混合)、`output`(Q6_K) 都不是 Q4_K。反量化已写好
   （`quant/ggml_blocks.h`），缺 SYCL 侧的点积。
2. **按层流式加载**：9.65 GiB 权重 > 8 GiB 设备预算（`profile::kDeviceBudget`），每层 ~350 MB 预取。
3. **MoE 专家调度**：按 `idx` 对 6 个 expert 调 gemv + 2 个 shared（GGUF 里是融合的 2816 宽）。
4. **V2-Lite 的 27 层 runner**：仿 `code/model/sycl/runner.cpp`，换掉注意力与 FFN 两处。
5. （可选）SoA 权重重排、32 B 载入、matrix-absorbed MLA —— 都排在 runner 之后。

## 5. 已知坑（都踩过，别再踩）

| 坑 | 症状 | 结论 |
|---|---|---|
| `quant` 在 SG=16 只量化半块 | int8 不一致 3327 / scale 差 1.2e-2 | 已修（`q8_block<SG>`），**任何 SG 都要跑 quant 抽检** |
| GEMV 的 `rel_l2` 是自洽的 | 用设备上的激活做参考，一起错也 PASS | 激活链要单独验（host 检查的 §0/§0b/§0c） |
| 目标机 `q4k_15b.gguf` 不是 Qwen2.5 | `general.name` = DeepSeek R1 Distill Qwen 1.5B **Abliterated**，词表没有 `<|im_start|>` | prompt 模板按词表自动选（chatml / r1 / none） |
| 特殊 token 被 BPE 拆开 | `<|im_start|>` → 6 个 token，输出乱码 | 已修（词表里 `<...>` 整体匹配），CUDA 路径同步修好 |
| `.bat` 是 LF 换行 | cmd 把 `build_sycl.bat` 吃成 `d_sycl.bat` | 已全转 CRLF + `.gitattributes` 锁住 |
| 目标机没有散热器 | 同一 kernel 冷/热/并发差 1.6 倍 | 见 §7 协议；`--ab-iters 2000`、`--ab-cooldown 0` |
| V2-Lite 的 FFN 宽度不是 256 倍数 | `10944`、`1408` 装不进 Q4_K | 上游用 Q8_0/Q5_0 解决；引擎要按张量分派量化类型 |
| 目标机 WinRM 间歇 | 长命令会超时 | 重试 2~3 次；或走 SMB 传文件 |

## 6. 大文件在哪（都不在仓库里）

| 文件 | 位置 | 大小 |
|---|---|---:|
| Qwen2.5-1.5B（`q4k_15b.gguf`） | `build\` | 0.81 GB |
| Qwen2.5-7B（`q4k.gguf`） | `build\` | 4.00 GB |
| Abliterated 1.5B（`abl15b_q4k.gguf`） | `build\` | 0.94 GB |
| **DeepSeek-V2-Lite Q4_K_M** | `D:\dsv2\`（主） + `E:\dsv2\`（副本） | 9.65 GB each |

> 本机 C: 只剩 **0.19 GB**。要腾空间就把 `build\*.gguf` 挪到 E:（11.9 GB 空闲），
> 但记得 `run_tests.bat` 里写死了 `build\q4k_15b.gguf`。

## 7. 目标机（小新）

- `192.168.137.8` / `Administrator`，口令见密码管理器（**仓库里不写明文**）。
- 路径、编译顺序、WinRM/SMB 用法、测量协议：**`HANDOFF_ARC.md`**。
- 现在的网络：WinRM 间歇可用；SMB 稳定；**目标机本身访问不了 HF/ModelScope**，模型要中转。

## 8. 仓库结构（本次整理后）

```
docs/
  STATE.md               ← 本文件（先读这个）
  HANDOFF_ARC.md         目标机交接：账号/路径/测量协议/坑
  PERF_BASELINE.md       性能基线与测量方法（§7 协议、§8 A/B、§10 quant bug、§11 门槛）
  SYCL_PORT.md           SYCL 后端总说明（§9 怎么验、§11 方案 C、§12 runner）
  DEEPSEEK_V2_LITE.md    V2-Lite 移植：形状、MLA/MoE、真实 GGUF 布局、缺什么
tools/
  gguf_dump.cpp          裸解析 GGUF 头（绕开类型白名单，截断文件也能跑）
  quant_dump.cpp         读真文件：类型直方图 + 每类型抽一行反量化 + 位打包自检
tests/
  sycl_bench.cpp         算子基准 + 四臂 A/B + 门槛
  sycl_gemv_host_check.cpp   CPU 垫片跑真 GEMV/quant/rmsnorm/swiglu
  sycl_dsv2_host_check.cpp   CPU 垫片跑真 MLA/MoE
  host_sycl/sycl/sycl.hpp    最小 SYCL 垫片（线程=work-item、条件变量=barrier）
build_sycl.bat / build_sycl_host.bat / build_sycl_e2e.bat / build_dsv2_host.bat
```
