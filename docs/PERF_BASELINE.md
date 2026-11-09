# 性能基准备忘录：Arc 130T / Core Ultra 5 225H

> 记录时间：2026-11-09。基线数字来自本机实跑，不是引用值：
> `tests/sycl_bench.cpp`（算子基准与数值对拍）、`tools/bench_ab.cpp`（512 MB 顺序扫描、USM A/B、热漂移）、
> `tools/dispatch_probe.cpp`（空 kernel 提交开销）。
> 编译器 Intel oneAPI DPC++/C++ 2026.1.1（`icpx`），系统 Windows 11。
> 本文是基线，不是目标；下一阶段见 `docs/SYCL_PORT.md` §11（方案 C）。

## 1. 硬件实测参数

| 项 | 实测值 | 说明 |
|---|---:|---|
| device | Intel(R) Arc(TM) 130T GPU (16GB) | 运行期 `DeviceCaps::report()` |
| Xe cores（EU 口径） | 112 | 运行期打印 |
| clock | 2200 MHz | 运行期打印 |
| global mem | 16831 MiB | 设备可见全局内存 |
| max alloc | **4095 MiB** | 单次分配上限 |
| local mem | **64 KiB** | 旧文档写 32 KiB，错；以实测为准 |
| max workgroup | 1024 | |
| subgroup 默认 | 8 | |
| subgroup 支持 | 8 / 16 / 32 | 本基线使用 16 |
| USM device | 支持 | `sycl::malloc_device` |
| USM shared | 支持 | `sycl::malloc_shared` |
| fp16 | 支持 | |
| oneAPI | 2026.1.1 | `icpx --version` |
| 驱动 / runtime | 1.13.35227 / 12.74.4 | 运行期打印 |

两条与代码常量不一致的地方，按实测记：

- `code/device/sycl/device.h` 的 `profile::kLocalMemBytes` 仍是 `32 * 1024`，运行期实测是 **64 KiB**。
  prefill 的 KV tile `static_assert` 还按 32 KiB 卡，属于保守值，不影响正确性。
- `profile::kXeCores` 写的是 8，运行期打印的是 `Xe cores(EU)=112`。这两个常量都不参与分支决策。

## 2. 当前基线

### 2.1 带宽与吞吐

| 项 | 值 | 口径 |
|---|---:|---|
| 硬件顺序读上限 | **61.38 GB/s** | `bench_ab`，512 MB 扫描，12 轮中位数 |
| GEMV 有效带宽 | **18.27 GB/s** | ROWS=8，SG=16，1.5B `blk.0.attn_q.weight` |
| 1.5B decode（推算） | **20.36 tok/s** | 18.27 GB/s /（权重 828 MiB + 每 token KV 28 MiB，ctx=512） |

GEMV 基准读的是 1.5B 的 `blk.0.attn_q.weight`（Q4_K，1536×1536，1.266 MiB）；
1.5B 的 tok/s 用的是全模型 828 MiB 权重，不是这个张量。20.36 tok/s 是纯带宽推算；
端到端按 71 ms/token 口径，见第 3 节的 dispatch 结论。

### 2.2 数值对拍

| 算子 | rel_l2 | 判定 |
|---|---:|---|
| GEMV | **8.437e-08** | PASS（门槛 < 1e-4） |
| attn_decode | **9.714e-08** | PASS |
| attn_prefill | **8.997e-08** | PASS |
| rms_norm | **7.568e-08** | ok（门槛 < 1e-5） |
| rope | **2.648e-08** | ok（门槛 < 1e-5） |

### 2.3 参数与平台实验

| 实验 | 结果 | 读法 |
|---|---|---|
| ROWS=4 | 18.15 GB/s | 略低于 ROWS=8 |
| ROWS=8 | 18.27 GB/s | 平台 |
| ROWS=16 | 14.89 GB/s | occupancy 下降，反而变差 |
| SG=16（计算类） | 18.05 GB/s | 比 SG=32 快约 56% |
| SG=32（计算类） | 11.56 GB/s | 对照 |
| SG=16 的 `attn_decode` | 2.1e38 | 数值爆掉，attention 必须钉在 32 lane |
| USM device | 61.38 GB/s | 与 shared 差 8.6% |
| USM shared | 56.54 GB/s | 标准差 9.8%，差异不显著 |
| 空 kernel dispatch | 约 5 µs/次 | submit-only |
| 热漂移前 1/3 | 53.40 GB/s | `bench_ab` 中位数 |
| 热漂移后 1/3 | 66.34 GB/s | +24%，是升频不是降频 |

ROWS 扫描与 SG 对比来自不同轮次，绝对值有波动，结论看排序。

## 3. 关键结论

1. **dispatch 不是瓶颈。** 空 kernel 约 5 µs/次；每层 14 次 × 28 层 = 1.96 ms/token，
   按 71 ms/token 的端到端总时间口径只占 2.7%。Kernel Fusion 不值得做。
2. **热漂移是升频，不是降频。** `bench_ab` 前 1/3 中位数 53.40 GB/s，后 1/3 66.34 GB/s，+24%。
   跑分必须先热身 3-5 轮再计时，否则测到的是低频态。
3. **USM 差异不显著。** device 比 shared 快 8.6%，但标准差 9.8%，在噪声内；保留 `malloc_device`。
4. **ROWS=8 已是平台。** 4/8/16 分别是 18.15/18.27/14.89 GB/s；再加行数只降 occupancy。
5. **SG 必须拆分。** 计算类内核 SG=16 比 SG=32 快约 56%，但 attention 按 32 lane 写死
   （`lane + 32*c`），SG=16 会把 attn 打到 2.1e38；最终 `kSyclSgCompute=16`、`kSyclSgAttn=32`。
6. **GEMV 只有顺序扫描的 30%。** 18.27 对 61.38；根因是跨步访存，不是访问宽度。方案 C 见第 4 节。

## 4. 已知瓶颈与下一目标

当前最大的瓶颈是 GEMV 的访存形状：18.27 GB/s 只有 512 MB 顺序扫描 61.38 GB/s 的 30%。
dispatch（2.7%）和 USM 选择（8.6%，不显著）都不是主因；ROWS 已经到平台，SG 已经拆到最优。

下一目标是方案 C：把「1 个 subgroup 扫一整行」改成「4 个 subgroup 分段扫同一行的 4 个连续段」，
让跨 subgroup 的地址连续，填满 L2 合并窗口。细节与风险见 `docs/SYCL_PORT.md` §11。

| 项 | 现状 | 目标 |
|---|---:|---:|
| GEMV 有效带宽 | 18.27 GB/s | 30+ GB/s（预期 26~30） |
| 1.5B decode（推算） | 20.36 tok/s | 25 tok/s |
| GEMV rel_l2 | 8.437e-08 | < 1e-4，不允许为速度牺牲 |

## 5. 复现方法

### 5.1 构建

与 `make.bat` 同一套环境：先载入 MSVC 与 oneAPI，再 `cd` 到仓库根目录。

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cd /d C:\nova
if not exist build\sycl mkdir build\sycl
icpx --version

icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel ^
     -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=0 ^
     -DNOVA_SYCL_SG=16 -DNOVA_SYCL_GEMV_ROWS=8 ^
     -Iinclude -Icode\device\sycl -Isrc\kernel\sycl ^
     code\device\sycl\device.cpp ^
     src\kernel\sycl\q4k_gemv.cpp ^
     src\kernel\sycl\q4k_gemv4.cpp ^
     src\kernel\sycl\quant.cpp ^
     src\kernel\sycl\norm.cpp ^
     src\kernel\sycl\rope.cpp ^
     src\kernel\sycl\elem.cpp ^
     src\kernel\sycl\attn.cpp ^
     src\loader\gguf.cpp ^
     tests\sycl_bench.cpp ^
     -o build\sycl\sycl_bench.exe
```

`C:\nova` 是目标机的仓库根目录，换机器时改成实际路径。
`NOVA_SYCL_DP4A=0` 是基线构建（纯标量）；`build_sycl.bat` 会先用 `dp4a_probe.cpp` 探硬件写法并注入 1/2，
两者只影响 dp4a 路径，不影响本文的带宽结论。`build_sycl.bat 1.5b` 也能编，但它不传 `NOVA_SYCL_SG`，
默认是 32，和本文的 SG=16 基线不同。

### 5.2 跑基准

```bat
cd /d C:\nova
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf
```

默认 `--iters 100 --warmup 5`。基线表用这个默认值；要压噪声用 `--iters 300`，
但不要把不同 `--iters` 的数混在一张表里。退出码 = 未通过的门槛数，0 = 全过。

读法：先看 §0 设备，再 §3 的 `rel_l2`，再 §4 的 `有效带宽`，最后 §7 的 `tok/s`。
改 kernel 后必须先看 `rel_l2`，再看带宽。

### 5.3 用 bench_ab 做 A/B

```bat
icpx -fsycl -O3 -std=c++17 tools\bench_ab.cpp -o build\sycl\bench_ab.exe
build\sycl\bench_ab.exe
```

它分配 512 MB device + 512 MB shared，预热一次、剔除第一轮，打印每轮值、中位数、标准差、
前 1/3 → 后 1/3 的漂移，以及显著性结论。基线输出是 device 61.38 GB/s、shared 56.54 GB/s，
漂移 53.40 → 66.34（+24%），差 8.6%，标准差 9.8%（在噪声内）。

它的 A/B 是 device vs shared。要对两版 kernel 做 A/B，把两版各编一个 exe，在同一会话里
按 A/B/A/B 交替跑 3 轮，比较各自的中位数；不要先跑完 A 再跑完 B，热漂移会把差值吃掉。
判据：差值小于 2 倍标准差就判为不显著（工具自己用的是这个规则）。

### 5.4 dispatch 探针

```bat
icpx -fsycl -O3 -std=c++17 tools\dispatch_probe.cpp -o build\sycl\dispatch_probe.exe
build\sycl\dispatch_probe.exe
```

看 `submit-only` 行（out-of-order 5.13 µs、in-order 4.97 µs）。`submit+wait` 的 13-15 µs
含同步，不是 dispatch 成本。

### 5.5 复现清单

1. `bench_ab`：device 中位数在 61 GB/s 量级，漂移为正；
2. `sycl_bench`：五项 `rel_l2` 都在 1e-7 量级；
3. 记录 GEMV 有效带宽与 1.5B tok/s；
4. 改 kernel 后重复 1-3，先对拍后看速度。

> 本文只写本机跑出来的数。没有实测的项（30 分钟持续负载、方案 C 的收益）不在这里下结论。
