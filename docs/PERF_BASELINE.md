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

> 2026-11-09 更新：方案 C 的代码已经落地（`NOVA_SYCL_GEMV_SPLIT`，默认值已按实测回退到 1），
> 数值在本机验完、带宽也在目标机测完 —— **方案 C 输了，见第 8 节**。测量协议上的坑见第 7 节（提频、并发、冷却）；
> `--warm-ms` / `--ab-iters` / `--ab-cooldown` 三个开关就是为它们加的。
> 下面这张「目标」表保持原样（它是当时的假设），但注意 **「现状」那一列本身就是假象**：
> 修正后的 GEMV 稳态是 36.8 GB/s，实际 decode 推算 41 tok/s，见第 7.5 节。

| 项 | 现状（旧口径，偏低） | 目标 |
|---|---:|---:|
| GEMV 有效带宽 | 18.27 GB/s → **实测 36.8** | 30+ GB/s（预期 26~30，已被旧映射达到） |
| 1.5B decode（推算） | 20.36 tok/s → **实测 41** | 25 tok/s |
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

## 6. 方案 C：代码已落地，目标机还没测

> **本节所有数字都是开发机（GTX 1080 Ti，无 Intel GPU、无 oneAPI）上跑出来的**：用
> `tests/host_sycl` 的最小 SYCL 垫片把 `src/kernel/sycl/q4k_gemv.cpp` 的 kernel 体直接跑在
> CPU 上，和「反量化 + double 累加」对拍。它只证明映射对、归约对，**不证明带宽**。
> 第 4 节表里的 26~30 GB/s / 25 tok/s 依然是待验证的假设。

### 6.1 改了什么

| 项 | 值 | 说明 |
|---|---|---|
| `NOVA_SYCL_GEMV_SPLIT` | **默认 1** | 一行由几个 subgroup 分段扫（1 = 旧映射）。实测 4 段更慢，默认已回退，见 §8 |
| `NOVA_SYCL_GEMV_STRIPE` | 默认 0 | 实验臂：4 个 subgroup 交错扫一行（段间不连续）。只在 A/B 里用 |
| `SPLITS=1` | 保留 | 原样基线，A/B 的对照臂 |
| gemv4 | 未改 | 仍是 1 subgroup = 1 行；`sycl_bench` 用「gemv4 第 0 列 vs 单列 gemv(SPLITS=1)」逐位钉住 |

`sycl_bench` 新增 §3b（四个变体逐个对拍）和 §4c（A/B：同进程交替 A/B/A/B × 3 轮、比中位数、
差值 < 2σ 判不显著）。第 5.3 节那个「先跑完 A 再跑完 B 会被热漂移吃掉差值」的坑，
现在由同一进程内交替来规避。

### 6.2 本机实测：数值（门槛 `rel_l2 < 1e-4`）

| 张量 | SG | SPLITS=1（基线） | SPLITS=2 | SPLITS=4 连续 | SPLITS=4 交错 |
|---|---:|---:|---:|---:|---:|
| 1.5B（K=1536, BPR=6） | 16 | 9.706e-08 | 9.257e-08 | 9.505e-08 | 9.147e-08 |
| 1.5B（K=1536, BPR=6） | 32 | 7.862e-08 | 8.339e-08 | 8.233e-08 | 7.886e-08 |
| 7B（K=3584, BPR=14） | 16 | 1.006e-07 | 9.641e-08 | 9.382e-08 | 9.412e-08 |
| 7B（K=3584, BPR=14） | 32 | 9.218e-08 | 9.264e-08 | 9.126e-08 | 8.998e-08 |

四个变体全 PASS，全部在 1e-7 量级，与第 2.2 节基线的 8.437e-08 同级。两种映射之间的
`max_abs` 差是 9.5e-07 ~ 1.9e-06，属于「求和次序不同」的量级，不是映射错。
尾部守卫也验了：`rows` 不是「每 WG 行数」的整数倍时，最后一个 group 里越界的行不会被写，
前 `rows-1` 行仍然全对。

复现（只需 MSVC，不需要 oneAPI）：

```bat
build_sycl_host.bat 1.5b 16
build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf
build_sycl_host.bat 7b 16
build\sycl_gemv_host_check_7b_sg16.exe build\q4k.gguf
```

### 6.3 目标机上要量的

`build_sycl.bat 1.5b` 编出来的二进制里就有全部四个臂，直接跑：

```bat
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf --iters 300
```

| 看什么 | 结论 → 动作 |
|---|---|
| §3b 四行的 `rel_l2` | 有一行 ≥ 1e-4 → 停下，先修映射，不要看带宽 |
| §4c「4 连续」vs 基线：差值 > 2σ 且更快 | 方案 C 成立 → 用默认 `SPLIT=4`，把第 4 节的「目标」换成实测 |
| §4c 不显著 | 合并窗口不是主因 → 回到 `SYCL_PORT.md` §7.1 的 32 B 载入 / SoA 重排 |
| §4c 显著更慢 | 先试 `-DNOVA_SYCL_GEMV_SPLIT=2`（7B 上不掉 PD），再不行回 1，并把 PD 损失一起记下来 |
| §4c 的带宽列落点 | 26~30 GB/s = 达到预期；18~22 = 只吃到一点点；< 18 = 退回去 |
| §7 `tok/s(热)` | 1.5B 从 20.36 往 25 走才算端到端兑现 |

### 6.4 已知代价（先写在前面，别指望线性的 1.4~1.6 倍）

1. **每 lane 在飞的载入数掉了**：`PD = min(4, NP)`，1.5B 上 NP 从 3 降到 1（48 B → 16 B/ lane），
   7B 上从 7 降到 2（64 B → 32 B）。grid 大了 4 倍（192→768 个 WG），但**驻留** work-item 数由
   硬件上限决定，所以 1.5B 上真正的在飞字节是降的。
2. **1.5B 是这套方案最不友好的形状**：BPR=6 切成 4 段是 1/2/1/2，段长 1 的 subgroup 有一半
   lane 组空转。7B 的 BPR=14 切成 3/4/3/4，浪费小得多。而 18.27 GB/s 恰好是在 1.5B 上量的。
3. **块头空洞还在**：128 B 之后那 16 B 的 `d/dmin/scales` 照样读，占 11% 的字节。

所以 4c 里那个 `SPLITS=2` 的臂不是凑数：7B 上它把 PD 保持在 4，只把「一个 WG 摸几行」从 8 改成 4。
如果 4 段输了而 2 段持平，主因就是第 1 条。

> 本文只写本机跑出来的数。没有实测的项（30 分钟持续负载、方案 C 的带宽收益）不在这里下结论；
> 第 6 节给的是本机实测的**数值**，不是带宽。

## 7. 测量状态的坑（2026-11-09 补，这一节的数字全是在目标机上复核出来的）

第 2.1 节那两个数（18.27 GB/s / 20.36 tok/s）**不是这台机器的 GEMV 上限**，而是「低频态」下测到的。
把状态控制住之后，同一个 kernel、同一份代码能跑出 26~32 GB/s。三条实测证据：

### 7.1 同一份 kernel、同一个张量，绝对值能差 2.3 倍

目标机上多次复核，同一份 `gemv` 代码、同一个 1.5B `blk.0.attn_q.weight`：

| 状态 | us/次 | GB/s | 出处 |
|---|---:|---:|---|
| 冷启动 + 5 次 warmup，且机器上还有 3 个并发编译 | 82.50 | 16.09 | 被污染的那轮（§7.4） |
| 冷启动，干净机器，第 1 批 20 次 | 49.3~51.0 | 26.0~26.9 | §4a 批 1 |
| 提频到第 8 批（累计 160 次发射） | 41.6 | 31.9 | §4a 批 8 |
| 烧 2 s 后短测（机器刚被三个 sweep 烧过） | 50.07 | 26.5 | 复核 A §4 |
| 烧 2 s 后短测（空闲机器） | 38.46 | 34.5 | 复核 B §4 |
| **2000 次长测量 × 3 轮（稳态）** | **36.07** | **36.79** | 复核 B §4c |

提频曲线（§4a，连跑 8 批 × 20 次）：50.98 → 47.17 → 47.12 → 47.03 → 47.33 → 47.30 → 43.67 → **41.66**，
批 8/批 1 = **+22.4%**。也就是说：只做 5 次 warmup（第 2.1 节的口径）等于在最慢的那一档上计时。

### 7.2 结论：只有「长测量 + 同轮交替」的表可信

短测量（10~300 次发射）落在提频/唤醒的过渡段，单次绝对值不可比；两次复核之间也差 25%
（50.07 vs 38.46 us，同样的协议，只是机器之前的热历史不同）。**所以：**

- 绝对值只有 `--ab-iters >= 2000`（≥70 ms）的稳态表才算数；
- 判方案优劣只看同一轮内的四臂相对差（下一节 §8 就是这么量的，σ < 1%）。

### 7.3 冷却反而是坑：短测量 + 空闲 = 测到 idle 时钟

「每轮之间 sleep 5-10 s」这条经验在**短测量**上是有害的：一次测量只有 10 次发射（0.5 ms）时，
6 s 空闲足以让 GPU 掉回 idle 时钟。实测同一个臂：轮 1 = 50.35 us，轮 2 = 19669.82 us（唤醒惩罚），
轮 3 = 19745.00 us；另外三个臂也从 ~51 us 掉到 ~100 us。

**正确做法**：一次测量至少 50~100 ms（`--ab-iters >= 2000`），四臂在同一轮内连续交替、
**不要**在臂与臂之间 idle。冷却只在「整轮之间、且整轮足够长」时才有意义。

### 7.4 并发是致命的

同一次复核里发现目标机上同时跑着三个批处理（`run_c.bat` / `run_c2.bat` / `c_sweep.bat`），
它们互相抢 CPU 与 iGPU 的功耗预算，那一轮的 GEMV 读数是 **16.09 GB/s**（干净环境下 26.5）。
判据：**同一时刻只允许一个测量进程**，编译也不要和测量并行。

### 7.5 所以第 2.1 节应该这样读

| 项 | 原文（低频态/受污染） | 复核（干净、长测量、稳态） | 读法 |
|---|---:|---:|---|
| GEMV 有效带宽 | 18.27 GB/s | **36.8 GB/s** | 是 61.38 的 **60%**，不是 30% |
| 1.5B decode（推算） | 20.36 tok/s | **41 tok/s**（36.8 GB/s 口径） | 「25 tok/s 目标」旧映射早就过了 |
| 方案 C 的收益空间 | 「26~30 GB/s 是目标」 | 目标区间低于旧映射的实测值 | 方案 C 已实测为负收益，见第 8 节 |

修正后的复测命令（`--warm-ms` / `--ab-iters` / `--ab-cooldown` 都已内建）：

```bat
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf --iters 300 --ab-iters 2000 --ab-cooldown 0 --warm-ms 2000 --no-cold
```

同一个二进制里两次 §4 的对比（说明「绝对值会漂，相对差不会」）：
`§4 首次 38.46 us` → `§4d 复测 44.06 us`，**+14.5%**，而同一时段的 §4c 四臂漂移只有 **-1.5%**。
所以：**结论只从 §4c 这种「同轮内交替、长测量」的表里读。**

> 教训一句话：在这台机器上，**「测什么」之前先问「芯片当时在什么状态」**。
> 同一个 kernel 在冷态、提频态、热稳态下能差 1.6 倍，比方案 C 想赚的那 6% 大一个数量级。

## 8. 方案 C 的 A/B 结论（实测）：旧映射最快，方案 C 显著更慢

目标机、1.5B、SG=16、`SPLITS=1/2/4连续/4交错` 四臂，同一进程内交替、每臂每次 2000 次发射
（≈72 ms）、3 轮、轮间不空闲（冷却会掉到 idle 时钟，见 §7.3）：

| 臂 | 轮1 | 轮2 | 轮3 | 中位数 | 带宽 | vs 基线 | σ | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **SPLITS=1（旧映射）** | 36.49 | 36.07 | 35.95 | **36.07 us** | **36.79 GB/s** | — | 0.9% | 基线 |
| SPLITS=2 连续段 | 42.22 | 41.60 | 41.99 | 41.99 us | 31.61 GB/s | **-14.1%** | 0.8% | 显著变慢 |
| SPLITS=4 连续段（方案 C） | 37.05 | 37.31 | 37.06 | 37.06 us | 35.81 GB/s | **-2.7%** | 0.5% | 显著变慢 |
| SPLITS=4 交错段 | 39.07 | 38.99 | 38.99 | 38.99 us | 34.04 GB/s | **-7.5%** | 0.1% | 显著变慢 |

降频检查：轮3/轮1 分别 -1.5% / -0.6% / +0.0% / -0.2% → **这次测量是稳态**，差值可信。
另一次独立运行（10 次发射的短测量）给出同样的**排序**（基线最好、2 段最差），只是绝对值整体偏高。
`rel_l2` 四个臂全部 PASS（8.4e-08 ~ 9.3e-08），所以这不是数值问题。

结论：

1. **方案 C 不成立。** 4 段连续 -2.7%（显著），交错 -7.5%，2 段 -14.1%。旧映射本来就是最优。
2. 主因就是 §6.4 那两条自己写下的代价：1.5B 上 `PD` 从 3 掉到 1（每 lane 在飞 48 B → 16 B），
   加上 BPR=6 切成 1/2/1/2 的段不均衡。合并窗口那点收益远不够补。
3. 更要紧的是**原来的前提本身是测量假象**：GEMV 不是「18 GB/s = 峰值的 30%」，而是
   「36.8 GB/s = 60%」。剩下 40% 的差距不再由「跨 subgroup 不连续」解释 —— 那个假设被这次
   A/B 直接否掉了。下一步要追的话，方向是 §7.1 的 32 B 载入 / 权重 SoA 重排，而不是继续调 lane 映射。
4. 默认值已回退到 `NOVA_SYCL_GEMV_SPLIT=1`（`sycl_common.h`）。方案 C 的代码和四个臂都留着，
   `-DNOVA_SYCL_GEMV_SPLIT=4` 可复现，但它不再是被推荐的配置。


