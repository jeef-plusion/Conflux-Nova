# 目标目录结构（V2.4.0 现状 + V2.5 物理迁移清单）

> 这份文件说明"引擎 / 应用 / 探针 / 基准"四块的**边界**，
> 以及为什么 V2.4.0 还没有把文件真正挪进这些目录。

---

## 1. 四块的边界（逻辑上已经分开）

| 目录（目标） | 放什么 | 现在实际在哪 |
|---|---|---|
| `core/` | 引擎底层：kernel、HAL/设备层、计算图。**不含任何 main()** | `src/kernel/sycl/`、`code/device/sycl/`、`code/graph/`、`code/model/sycl/` |
| `apps/` | 端到端启动器（CLI / server） | `code/main.cpp`（CUDA 辩论演示）、`tests/sycl_e2e.cpp`（SYCL CLI） |
| `novascope/` | 探针、仪表盘、以及 D3D12/WDDM 这些**失败实验** | `code/model/*/scope_*.h`、`tools/novascope.html`、`tools/descope*.cpp`、`tools/ghost_pool.cu`、`tools/d3d12_steal.cu`、`tools/cpu_v_gemv.cpp`、`tests/*scope*.cpp` |
| `bench/` | 微基准工具 | `tests/sycl_bench.cpp`、`tests/sycl_gemv_host_check.cpp`、`tests/sycl_dsv2_host_check.cpp`、`tests/sycl_soa_bench.cpp` |

**关键约束（已满足）**：`core` 的编译配置里没有探针代码。

- `build_sycl.bat` / `build_sycl_e2e.bat` 的源文件清单里**不含**任何 `scope_*.h` 的 .cpp；
  探针是**头文件**（`scope_core.h` / `scope_sycl.h`），只有 runner/device 在编译期
  `#include` 它，而 `NOVA_SCOPE` 未定义时 `NOVA_SCOPED` 宏展开为原语句 —— 零代码、零 dll。
- 仪表盘、D3D12/WDDM 实验、CPU GEMV 基准都是**独立可执行文件**，不参与引擎编译。

## 2. 为什么 V2.4.0 没做物理移动

12 个编译脚本硬编码了路径（`-Isrc/kernel/sycl`、`code/device/sycl/...`、`tests/sycl_e2e.cpp`），
物理移动必须同步改这些脚本 + 所有 `#include`，一次改错就会让"已验证的 V2.4.0"编译不过。
在**发版节点**上冒这个风险不值得，所以拆成两步：

- **V2.4.0（本次）**：边界文档化 + 发行包 + 版本标记；逻辑边界已经清晰（见上表）。
- **V2.5（下一步）**：物理迁移，有明确清单和验收标准。

## 3. V2.5 迁移清单（照做即可）

1. `git mv src/kernel core/kernel`、`git mv code/device core/device`、`git mv code/graph core/graph`、
   `git mv code/model core/model`；
2. `git mv code/main.cpp apps/cuda_debate.cpp`、`git mv tests/sycl_e2e.cpp apps/sycl_cli.cpp`；
3. `git mv` 探针与实验到 `novascope/`（见 §1 第 3 行清单），并把 runner/device 的
   `#include "model/sycl/scope_sycl.h"` 改成 `"novascope/scope_sycl.h"`；
4. `git mv tests/sycl_bench.cpp bench/`、`tests/sycl_*_host_check.cpp bench/`；
5. 编译脚本改为：`-Icore -Iinclude -Iapps`，源文件列表按新路径；
6. **验收**：`build_sycl_host.bat 1.5b 16` → `失败项 = 0`；`pd_sweep.bat 4` → 27.5 ms/token；
   `nova --tokens 8` → 正常出字。

## 4. 发行包（`release/`）

二进制**不进 git**（`.gitignore` 里排除了 `release/*.exe`、`release/*.dll`），
只提交三样：`run_bench.bat`、`assemble.bat`、`QUICKSTART.md`。
装配：`release\assemble.bat`（在装好 oneAPI 的机器上跑一次）→ 30 MB 绿色包 → 传 GitHub Releases。
