# BUILD · 三分钟构建

> **平台：Windows x64 only。** 本项目只在 Windows 上验证过（CUDA = MSVC + nvcc，
> SYCL = oneAPI + Intel 显卡驱动）。没有 Linux/macOS 构建脚本，也不假装支持。
>
> 一套 CMake，三个后端选项。根目录那 22 个 `.bat` 是历史遗留的快捷方式，
> **新代码与 CI 只走 CMake**。

---

## 1. 最短路径

```bat
build.bat                    :: CUDA + Qwen2.5-7B（默认）
build.bat sycl 1.5b          :: Intel Arc + Qwen2.5-1.5B
build.bat host               :: 只编 CPU 垫片（任何机器、无显卡）
```

等价的手工命令：

```bash
cmake -B build-cmake -DNOVA_BACKEND=CUDA -DNOVA_MODEL=QWEN25_7B
cmake --build build-cmake -j
ctest --test-dir build-cmake --output-on-failure
```

---

## 2. 选项表

| 选项 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `NOVA_BACKEND` | `CUDA` / `SYCL` / `HOST` | `CUDA` | HOST 只编 CPU 垫片，CI 与无显卡机器用 |
| `NOVA_MODEL` | `QWEN25_7B` / `QWEN25_1P5B` / `DEEPSEEK_V2_LITE` | `QWEN25_7B` | 一个模型一个二进制（宏与旧 `.bat` 同名） |
| `NOVA_CUDA_ARCH` | `sm_61` / `sm_75` / `sm_86` / `native` | `sm_61` | **必须 ≥ sm_61**：`__dp4a` 只在 6.1+ 存在 |
| `NOVA_BUILD_APPS` | ON / OFF | ON | 端到端 CLI |
| `NOVA_BUILD_BENCH` | ON / OFF | ON | 基准与 CPU 垫片对拍 |
| `NOVA_ENABLE_SCOPE` | ON / OFF | ON | NovaScope 探针（纯头文件；OFF 时零代码） |

---

## 3. 依赖

| 后端 | 需要 | 验证 |
|---|---|---|
| CUDA | CUDA Toolkit 11.8+（`nvcc`）、MSVC 2022 或 gcc 11+ | `nvcc --version` |
| SYCL | Intel oneAPI（`icpx`）+ Intel 显卡驱动 | `icpx --version` |
| HOST | MSVC 或任意 C++17 编译器 + CMake 3.20+ | `cmake --version` |

SYCL 在 Windows 上：先 `call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"` 再跑 `build.bat sycl`。

> ✅ **SYCL 已通过 CMake 验证**（Arc 130T，2026-11-11）：`build.bat sycl 1.5b clean` → 52 s →
> `build-cmake\SYCL\nova_sycl.exe` → **27.25 ms/token = 36.70 tok/s**。
>
> 它有两个硬前提，`build.bat` 已代劳；手工跑 cmake 时必须自己保证：
> 1. **先 `vcvars64.bat` 再 `setvars.bat`**（反了 icpx 找不到 MSVC 的 C++ 标准库头，报 `type_traits not found`）；
> 2. **必须用 Ninja 生成器**（`-G Ninja`）：MSBuild 会剥掉 Intel 的 LIB 路径，链接器报 `LNK1104: libmmd.lib`。

---

## 4. 产物

| 目标 | 后端 | 说明 |
|---|---|---|
| `nova_core(.exe)` | CUDA | 端到端（7B/1.5B 由 `NOVA_MODEL` 决定） |
| `nova_sycl(.exe)` | SYCL | 端到端 decode + 采样 |
| `nova_host_check(.exe)` | 任意 | **数值闸门**：真内核跑在 CPU 垫片上对 double 参考 |

**发布前必须过的一关**：

```bash
nova_host_check build/q4k_15b.gguf    # 期望最后一行：失败项 = 0
```

---

## 5. 三个已知的构建坑（都踩过，已修）

1. **CUDA 架构必须先于 `enable_language(CUDA)` 设定** —— 否则 CMake 先把它初始化成 52，
   `__dp4a` 直接报 `identifier __dp4a is undefined`。
2. **MSVC 必须 `/utf-8`** —— 源码里有中文注释与格式串，按 936 代码页解析会让
   `snprintf` 误报"参数不足"，运行期输出乱码。注意 `/utf-8` 只能给 C/C++，不能给 nvcc。
3. **`.bat` 必须 CRLF** —— 转成 LF 后 `goto` 标签会失效、构建静默失败。
   `.editorconfig` 已经把这条钉死。

---

## 6. 逐步：从零到跑分（1.5B on Arc）

```bat
:: 1) 拉代码
git clone <repo> && cd nova_v2

:: 2) 模型（约 960 MB，不进 git）
::    放到 build\q4k_15b.gguf

:: 3) 构建
build.bat sycl 1.5b

:: 4) 数值闸门
nova_host_check build\q4k_15b.gguf

:: 5) 跑分（实测 34–36 tok/s，视机器与散热）
build-cmake\Release\nova_sycl.exe --gguf build\q4k_15b.gguf --tokens 32 --warmup 2
```

> 想要"双击就跑"的绿色包？见 [`release/QUICKSTART.md`](release/QUICKSTART.md)。
