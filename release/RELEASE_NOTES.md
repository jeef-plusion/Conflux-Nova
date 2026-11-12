# FJEC_V2.4.0 · 发版说明

> 仓库：`jeef-plusion/Conflux-Nova` ｜ Tag：`v2.4.0-release` ｜ Commit：`cdfddc3`
> 平台：**Windows x64**（CUDA / SYCL / HOST 三后端，均已实测）

---

## 标题

**FJEC_V2.4.0: 纯血推理引擎正式发布（Windows x64 · CUDA + SYCL 双后端）**

> 备选（若想强调单平台诚实性）：
> `FJEC_V2.4.0: 双后端推理引擎正式发布 —— CUDA / SYCL / HOST 一条命令构建`

---

## 一句话

一颗 Intel Arc 130T 核显，跑 1.5B 模型 **36.70 tok/s**（27.25 ms/token），GEMV 带宽
**37–38 GB/s**（顺序读上限实测 61.38 GB/s 的 62%），并且**三个后端都能用一条命令编出来**。

---

## 核心卖点

### 1. 三端统一构建：CMake + CI 全面落地

一套 `CMakeLists.txt` 取代了根目录 23 个 `.bat`：

| 后端 | 命令 | 耗时 | 产物 | 状态 |
|---|---|---|---|---|
| CUDA | `build.bat cuda 7b` | ~60 s | `nova_core.exe` | ✅ 实测 |
| SYCL | `build.bat sycl 1.5b` | **52 s** | `nova_sycl.exe` | ✅ 实测 36.70 tok/s |
| HOST | `build.bat host 1.5b` | ~40 s | `nova_host_check.exe` | ✅ 数值闸门 `失败项 = 0` |

选项全部显式：`-DNOVA_BACKEND=CUDA|SYCL|HOST`、`-DNOVA_MODEL=QWEN25_7B|QWEN25_1P5B|DEEPSEEK_V2_LITE`、
`-DNOVA_CUDA_ARCH=sm_61|sm_75|sm_86|native`、`-DNOVA_ENABLE_SCOPE=ON|OFF`。
GitHub Actions 三道闸门：clang-format、HOST 数值对拍（Windows）、CUDA 编译。
**跑不了的闸门会明说跳过，不装绿。**

### 2. 性能怪兽

| 指标 | 数值 | 口径 |
|---|---|---|
| 端到端 decode（1.5B，Arc 130T） | **27.25 ms/token = 36.70 tok/s** | 16 token，预热 2，单进程 |
| GEMV 带宽（down / gate / up） | **37.4 / 38.1 / 37.8 GB/s** | 每算子有效带宽 |
| 顺序读上限（同机实测） | 61.38 GB/s | 基准线 |
| GPU 忙比 | **97.6%** | 无发射空洞 |

演进曲线：V2.3.0-alpha 的 14–18 GB/s → V2.4.0 的 37–38 GB/s（**+113%**）。
关键是修掉了那个致命的 quant bug（SG=16 时只量化半个 32 元素块），以及重定测量协议。

### 3. 架构解耦：文书层黑盒契约

- `core`（kernel / HAL / graph / model）不依赖任何上层概念；
- `apps` 只做端到端编排，不含算子；
- 探针是**纯头文件**：`NOVA_ENABLE_SCOPE=OFF` 时零代码、零运行时开销；
- 接口契约写进 `docs/LAYOUT.md`，跨后端差异（CUDA warp vs SYCL subgroup）逐条标注在源码注释里。

### 4. 极致运维：NovaScope

- 每个算子的设备起止时间、每次 H2D/D2H 的字节数、每层流量、GPU 忙比；
- 输出 Chrome/Perfetto 可吃的 trace JSON；
- 单文件 HTML 交通图仪表盘（粒子流 + 红绿拥堵 + Top-5 条形）；
- **诚实口径**：EU 利用率 / 频率 / 温度一律标 `n/a`（Windows + iGPU 没有可靠公开接口），绝不编数。

---

## 附件（预编译包，约 30 MB）

| 文件 | 说明 |
|---|---|
| `nova_sycl.exe` | SYCL 端到端 decode（真 28 层、真采样），0.69 MB |
| `sycl9.dll` / `libmmd.dll` / `svml_dispmd.dll` | Intel SYCL 运行时 |
| `ur_loader.dll` / `ur_win_proxy_loader.dll` | Unified Runtime 加载器 |
| `msvcp140.dll` / `msvcp140_1.dll` / `vcruntime140.dll` | MSVC 运行时 |
| `run_bench.bat` | 一键跑分（自动找模型、自动 call setvars） |
| `QUICKSTART.md` | 三步上手 + 5 条 FAQ |
| `assemble.bat` | 在本地重新装配这个包 |

**使用**：解压 → 把 `q4k_15b.gguf` 放进同目录 → 双击 `run_bench.bat`。
需要 Intel Arc 驱动；装了 oneAPI 的机器会自动使用它。

---

## 已知限制（诚实清单）

1. **平台只支持 Windows x64**：没有 Linux/macOS 构建脚本，也不声称支持；
2. **SYCL 需要 Intel Arc 显卡 + 驱动**；`build.bat sycl` 有两个硬前提（`vcvars64` → `setvars`
   的顺序、必须用 Ninja 生成器），脚本已代劳，手工 cmake 需自己保证，见 `BUILD.md` 第 3 节；
3. **DP4A 未启用**：SYCL 2020 没有 int8×int8→int32 的 `dot` 形式，硬件加速路径待用 Intel
   math 扩展或内联汇编补上；
4. **DeepSeek-V2-Lite** 的 MLA/MoE 算子已通过数值验证，但按层流式加载与 Q5_0/Q8_0/Q6_K
   的 GEMV 尚未完成，`docs/DEEPSEEK_V2_LITE.md` 有完整清单；
5. 1080 Ti 的 CUDA 路径被 WDDM 的 ~2.25 GiB 会话级显存预算卡住（llama.cpp 同样撞墙），
   相关取证在 `docs/NOVASCOPE.md` §7–§10。

---

## 三条格言（都踩出来的）

1. **测量协议比优化重要** —— 那个 18 GB/s 的"基线"是冷启动 + 并发造出来的假象；
2. **流量账算对了，收益也可能是零** —— 削掉 25% 流量只涨 3.4%，因为瓶颈在指令发射；
3. **不可能的前提用一行代码就能验伪** —— `sycl::dp4a` 这个 API 根本不存在，
   那条"硬件加速路径"写了几个月、从没编译过；
4. **没进编译清单的代码等于没写；没在产线上量的数字等于没测** ——
   这条让一次"36.32 tok/s"的发版数字改正成了 34.2–36.3 的范围。

---

## 验证命令

```bat
:: 数值闸门（不需要显卡）
build.bat host 1.5b
build-cmake\HOST\Release\nova_host_check.exe build\q4k_15b.gguf     :: 期望：失败项 = 0

:: 性能（Arc 机器）
build.bat sycl 1.5b
build-cmake\SYCL\nova_sycl.exe --gguf build\q4k_15b.gguf --tokens 32 --warmup 2
```
