# QUICKSTART · 三十秒跑出成绩

> 不需要装 Visual Studio，不需要装 oneAPI 编译器，不需要懂 C++。
> 只要：**Windows 11 + Intel Arc 显卡（已装官方驱动）+ 一个 GGUF 模型文件**。

---

## 一、三步跑起来

1. **准备模型**：把 `q4k_15b.gguf`（Qwen2.5-1.5B 的 Q4_K 量化，约 960 MB）
   放进本目录，或者改名成 `model.gguf` 也放进来。
   > 没有模型？用 llama.cpp 的 `quantize` 把任意 Qwen2.5-1.5B 转成 `q4k`，或用
   > `huggingface-cli download` 拉一个现成的 Q4_K GGUF。

2. **双击 `run_bench.bat`**。

3. **看最后几行**。长这样：

```
[e2e] 生成 32 个 token：27.54 ms/token -> 36.32 tok/s（decode，不含 prompt）
```

`36.32 tok/s` 就是这台机器上的**端到端 decode 速度**。跑完窗口不会关，方便截图。

---

## 二、模型不在这个目录怎么办

三种都行：

```bat
:: 1) 拖拽：把 gguf 文件直接拖到 run_bench.bat 上
:: 2) 命令行
run_bench.bat D:\models\q4k_15b.gguf
:: 3) 环境变量（高级）
set MODEL=D:\models\q4k_15b.gguf && run_bench.bat
```

---

## 三、常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 双击闪一下就没了 | 本目录缺 `sycl9.dll` / `libmmd.dll` / `svml_dispmd.dll`（发行包应自带）；或没装 Intel 显卡驱动 |
| 报 `No device found` / `找不到设备` | 显卡驱动太旧，或这台机器的 Arc 被禁用；装最新 Intel Arc 驱动 |
| 报 `cannot open` | 模型路径不对（见上面第二节） |
| 速度明显偏低（<20 tok/s） | 笔记本在省电模式、或同时跑了别的 GPU 任务；插电、关掉浏览器硬件加速再试 |
| `0xC0000135` | 缺 MSVC 运行时；安装 [VC++ 2015-2022 运行库 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

---

## 四、这些数字是怎么来的（想深究的话）

- 端到端包含：28 层 × 14 个算子 + 词表投影 + 采样 + 每 token 一次 608 KB 的 logits 回传；
- 实测口径是**连续生成 32 个 token 的平均值**，不含首次加载（约 0.9 s）与预热；
- 这颗 Arc 130T 的**顺序读上限实测 61.38 GB/s**，GEMV 实测 37–38 GB/s；
- 完整性能档案（含 4 种 lane 映射的 A/B、预取深度扫描、SoA 布局的失败记录）在
  [`docs/PERF_BASELINE.md`](../docs/PERF_BASELINE.md) 与 [`docs/SYCL_PORT.md`](../docs/SYCL_PORT.md)。

---

## 五、文件清单

| 文件 | 作用 |
|---|---|
| `nova_sycl.exe` | SYCL 端到端 decode（真 28 层、真采样） |
| `sycl9.dll`, `libmmd.dll`, `svml_dispmd.dll` | Intel SYCL 运行时（已随包附带） |
| `msvcp140.dll`, `vcruntime140.dll`, `msvcp140_1.dll` | MSVC 运行时（若系统已有可删） |
| `run_bench.bat` | 一键跑分 |
| `QUICKSTART.md` | 本文件 |
