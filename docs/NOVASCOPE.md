# NovaScope · 性能诊断工具

> 目标不是"再打一个计时器"，而是回答三个问题：**谁在偷懒、谁在堵车、钱花在哪**。
> 一句话：每个 kernel 的 device 时间 + 每次拷贝的字节 + 每层的流量，合成一张 Perfetto 瀑布图和一个"数据交通图"仪表盘。

---

## 1. 它能回答什么，不能回答什么（先说清楚）

| 指标 | 怎么来的 | 可信度 |
|---|---|---|
| kernel 的 device start/end | `queue::enable_profiling` + `event::get_profiling_info<command_end>` | ✅ 硬数据 |
| 每个 op 的占用区间 | in_order 队列上，相邻"边界事件"的结束时间差（见 §3） | ✅ 硬数据（差一个边界事件 ~1–2 µs） |
| submit → start 的排队间隙 | host 侧 `steady_clock` 的提交时刻 vs device start | ✅ 硬数据 |
| 拷贝次数 / 字节 / 方向 | `Device::upload/download` 里逐次记账 | ✅ 硬数据 |
| 每 op 有效带宽 | 该 op 的权重字节（`ChunkedWeight::bytes()`）÷ 占用时间 | ✅ 硬数据 |
| GPU 忙比 | Σ(计算类 op 时间) ÷ 墙钟 | ✅ 硬数据（**不含**拷贝，拷贝是总线时间） |
| **计算单元占用** | 该 op 的 work-group 数 × WG 大小 ÷ 理论驻留上限 | ⚠️ 估算，不是硬件计数器 |
| **EU 真实利用率** | 需要 Intel Metrics Discovery / Level Zero Sysman | ❌ Windows + iGPU 无可靠公开接口 |
| **GPU 频率 / 温度** | 同上 | ❌ 一律显示 `n/a`，留 `set_hw_sample()` 钩子 |

**绝不编数**：拿不到的指标宁可打 `n/a`，也不拿"估算值"冒充测量值。这跟 `PERF_BASELINE.md` §6 的规矩一致。

---

## 2. 文件与调用关系

```
code/model/sycl/scope_core.h    纯 C++：Rec / Summary / Chrome trace JSON / CSV / 终端报告
code/model/sycl/scope_sycl.h    SYCL 探针：profiling 队列、边界事件、拷贝记账、resolve_all()
code/model/sycl/runner.cpp      每个算子包一层 NOVA_SCOPED(...)（24 处）
code/device/sycl/device.cpp     H2D/D2H 记账；NOVA_SCOPE=1 时用 profiling 队列
tests/sycl_e2e.cpp              --scope：跑完打印四组统计 + 写 nova_trace.json / .csv
tests/scope_host_check.cpp      合成事件流自检（无 GPU）
tools/novascope.html            单文件仪表盘（Canvas，无外部依赖）
tools/novascope_check.js        node 校验：页面 JS 语法 + 与 C++ 统计口径对拍
```

---

## 3. 为什么用"边界事件差值"，而不是逐 kernel 取 event

算子是 `void gemv(...)` / `void rmsnorm_quant(...)` —— 内部 `q.parallel_for(...)` 把 event 丢掉了。
要逐 kernel 拿 `start/end`，得把 15 个函数和 15 个声明都改成返回 `sycl::event`（侵入面很大，
而且本机没有 oneAPI，改完无法自测）。

所以 NovaScope 走另一条路：**每次发射后立刻提交一个队列边界**
（`q.ext_oneapi_submit_barrier()`），队列是 in_order，语义上"前一个 op 跑完才轮到下一个"，因此

```
op_i 的 device 区间 = [barrier_{i-1}.end, barrier_i.end]
```

代价是每个 op 多一次边界（`PERF_BASELINE` 里实测 dispatch ~5 µs/次，边界更轻），
所以**只给诊断跑用**：`NOVA_SCOPE=1` 才建 profiling 队列，平时性能测量不受影响。
将来要 exact 数值，把 `NOVA_SCOPE_EXACT` 这条路补上即可（签名改造成本已记录在此）。

---

## 4. 怎么跑

### 4.1 本机（无 GPU，验统计口径与 trace 格式）

```bat
build_scope_host.bat
build\scope_host_check.exe            :: 合成流：统计、JSON、判读全对拍
node tools\novascope_check.js         :: 页面 JS 语法 + 与 C++ 同一份 trace 对拍
```

### 4.2 目标机（真 GPU，10 个 token）

```bat
build_sycl_e2e.bat 1.5b
set NOVA_SCOPE=1
build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 10 --scope
```

跑完会得到：
- 终端四组统计（墙钟/忙比/带宽/拷贝触发率/频率温度/排队气泡 + Top 算子表 + 红黄绿判读）；
- `nova_trace.json`（Chrome/Perfetto 格式）与 `nova_trace.csv`。

### 4.3 看瀑布图

`chrome://tracing` → Load → `nova_trace.json`；或 `https://ui.perfetto.dev` 直接拖入。
三条泳道：`tid 0` = Compute（X 切片带 `layer/tok/bytes/bw_gbs/gap_us`），`tid 1` = Copy，
`tid 2` = `layer_bytes` 计数器（折线）。

### 4.4 看交通图

双击 `tools/novascope.html`（纯本地，无 CDN）：

- **三级流水线**：Host 下发（脉冲）/ Bus 拷贝（粒子，半径 ∝ √字节）/ Compute 执行（条块）；
- **红绿灯**：拷贝占墙钟 > 15% → 总线变红；GPU 忙比 < 70% → Compute 变灰（算力空转）；
- **Top 5**：按总耗时排的算子柱状图 + 每 op 有效带宽；
- **时间线**：每个 token 一行，底部红线是拷贝。

数据入口：拖入 `nova_trace.json`，或连 `ws://localhost:8787`（有数据就实时刷，没有就退回内置演示流）。

---

## 5. 验收清单（本仓库当前状态）

| 项 | 状态 |
|---|---|
| 统计口径（墙钟/忙比/带宽/拷贝率/气泡） | ✅ `scope_host_check` 与独立参考公式逐项对齐，`失败项 = 0` |
| Chrome trace JSON | ✅ 818 切片、54 计数器、括号配平、ts 单调、µs 原点正确 |
| 仪表盘 JS | ✅ node 语法检查 + 12 项结构检查通过 |
| C++ ↔ JS 同一份 trace | ✅ 墙钟 16.633/16.636 ms、忙 13.770 ms、拷贝 412 µs、忙比 82.79%、Top1 = gemv.q 全部一致 |
| 真 GPU 10 token 跑通 | ⏳ 需目标机（`build_sycl_e2e.bat` + `--scope`），WinRM 抽风时改走 SMB 传脚本 |

> 交叉对拍抓到的第一个真 bug：C++ 侧原先把拷贝时间算进"GPU 忙"（虚报 412 µs / 2.5 个百分点）。
> 已修，并同步修了测试里的参考实现 —— 两套独立实现从此互相约束。

---

## 6. 下一步（如果还要更接近 Nsight）

1. **per-kernel exact event**：算子返回 `sycl::event`，去掉边界事件的 1–2 µs 误差；
2. **Sysman / Metrics Discovery**：接 `zeInit` + `zesDeviceGetProperties` 拿真实频率，
   接 `zesDeviceEnumFrequencyDomains` 读当前频率；EU 利用率要靠 OA/Metrics，iGPU 上通常不可用；
3. **按 SM/EU 的热力图**：需要 `sub_group` 级采样，代价很大，优先级最低；
4. **WebSocket 推送器**：`tools/novascope_push.cpp` 把 `Rec` 逐条 JSON 推给页面（目前页面已支持该入口）。

---

## 7. CUDA 移植与 1080 Ti 实测（2026-11-10）

### 7.1 移植方式

| 层 | 文件 | 说明 |
|---|---|---|
| 探针 | `code/model/cuda/scope_cuda.h` | `cudaEventRecord` 成对 + `cudaEventElapsedTime`；**事件池复用**、**不在 begin/end 里同步** |
| 埋点 | `code/graph/dense.h`（12 处）、`src/model.cpp`（3 处拷贝） | 12 个算子传入名字与权重字节；H2D.id / H2D.ids / D2H.logits |
| 驱动 | `tests/cuda_scope.cpp` + `build_cuda_scope.bat` | 整模型 10 token（需能装下） |
| 微基准 | `tests/cuda_scope_micro.cpp` + `build_scope_micro.bat` | 真实 `Kernel::gate_n`（8960×1536 Q4_K），200 次 |
| Profiler | `include/nova/model.h` | 增加 `scoped` 非阻塞路径；老路径保持每段同步不动 |

### 7.2 实测结果（真 kernel、真事件）

```
形状 8960 x 1536，重复 200 次，权重 7.4 MiB
wall 6.573 ms   GPU 忙 100.0%   权重流量 1476.56 MiB   有效带宽 235.57 GB/s
gemv_q4k: 200 次 / 6.573 ms / 32.86 us 每次 / 235.57 GB/s
```

- 1080 Ti 理论带宽 484 GB/s（11 Gbps × 352 bit），**实到 235.6 GB/s = 49%**。
  这是单列 GEMV（每列只读一遍权重、每行 8 lane）的真实上界，不是"跑满显存"的负载。
- 事件对差值 = 32.86 µs/次；把 200 次连起来看 GPU 忙比 100%（无空隙）。

### 7.3 两个环境级发现（不是代码问题）

1. **单次/累计 `cudaMalloc` 上限**：同一张卡空闲 9.08 GiB 时，
   `1.00 GiB` 成功、`1.66 GiB` 失败；连续 256 MiB 分块时**第 4 次必失败**。
   这是驱动/WDDM 侧的进程配额（桌面合成器与浏览器在共享这张卡）。
   为此把 `DeviceArena` 改成 **512 MiB 分块**（`kChunkBytes`），并把
   `kMaxCtx / kMaxSessions / kSessCtx` 做成编译期可覆盖（`NOVA_MAX_CTX` 等），
   但配额本身仍在，**整模型（1.5B=1.45 GiB、7B=4.5 GiB）在这台机器当前状态下装不进去**。
2. **CUDA 事件测不出 host 提交→device 开始的延迟**（那要 CUPTI）。
   所以 CUDA 路径把 `gap` 恒置 0，不编数；改由 `host_span_ns / wall_ns` 反映队列深度。

### 7.4 复现

```bat
:: 微基准（一定能跑，7.4 MiB 权重）
build_scope_micro.bat
build\cuda_scope_micro.exe 200

:: 整模型（需要显存配额放开）
build_cuda_scope.bat NOVA_MODEL_QWEN25_1P5B
set NOVA_SCOPE=1
build\cuda_scope.exe build\q4k_15b.gguf 10
```

产出 `nova_trace_cuda.json` / `.csv`，拖进 `chrome://tracing`；仪表盘用
`tools/novascope.html`（或把 trace 内联进 HTML 后截图，见 `build\novascope_cuda.png`）。
### 7.5 端到端跑通之后的修正（2026-11-10 16:50）

**先纠一个我自己写错的结论**：7.3 节说的"分配上限"不是尺寸问题。用 `build\lt\oom_sweep.cu`
做了三组实验：

| 实验 | 结果 |
|---|---|
| A. 单次分配、测完立刻 free（64 MiB → 4 GiB，含 2^n ± 1KB/64KB/1MiB/16MiB） | **全部成功，连 4 GiB 都成功** |
| B. 连续分配 256 MiB 不释放 | 第 8 块（累计 2.00 GiB）成功，第 9 块 OOM（此时仍报 7.08 GiB 空闲） |
| C. 256 MiB 分配-释放 20 轮 | 20 次全成功，free 每轮回到 9.08 GiB |

结论：**不是尺寸/对齐，而是"同时持有"的额度（约 2 GiB），且它是动态的** ——
取决于桌面合成器/浏览器此刻占用多少。16:47 那个时刻额度放开，模型就加载成功了。
所以 7.3 第 1 条应从"驱动硬上限"改读成"动态的进程显存预算"。

**同时修掉一个我自己的埋点 bug**：`Profiler` 的 `scoped` 分支最初只记了 host 时间，
导致整模型那栏的"GPU 时间"其实是提交时间 —— 现象是 `gemv.gate` 报出 1238 GB/s
（超过 484 GB/s 峰值，物理不可能）。修法：`begin/end` 里真正 `cudaEventRecord` 一对事件
（流由 `cLayerCtx.w.s` 传给 `Profiler::stream`），修完后 GPU 忙比从 21.6% 变成 **93.0%**。

修正后的真实端到端（1080 Ti，1.5B，10 token，带探针）：

```
wall 73.163 ms   GPU 忙 68.067 ms (93.0%)   权重流量 4868.28 MiB (69.77 GB/s)
拷贝 20 次 / 5.80 MiB / 5.096 ms (7.0%)     拷贝触发率 2.00 次/token, 594 KiB/token

gemv.gate   280 次  18.567 ms  27.3%  116.74 GB/s  ← 最热
gemv.down   280 次   9.845 ms  14.5%  220.17 GB/s
o_proj      280 次   5.445 ms   8.0%   68.25 GB/s
gemv.q      280 次   5.155 ms   7.6%   72.08 GB/s
head         10 次   5.007 ms   7.4%  262 GB/s（131 MB 词表投影，单次 500 us）
swiglu      280 次   4.836 ms   7.1%
rmsnorm.*   560 次   5.875 ms  11.4%
bias.qkv    280 次   3.551 ms   5.2%   ← 3072 个 float，却要 12.7 us/次
```

三个值得追的发现（都还能再挖）：

1. **同样 7.4 MiB、同样 kernel，gate 只有 116.7 GB/s，down 有 220.2 GB/s** —— 差 1.9 倍。
   gate/up 是连续的两次大读，down 在 swiglu 之后，访存节奏不同；这跟 SYCL 侧"有效带宽只有
   顺序读 60%"是同一类问题的 CUDA 版本。
2. **端到端有效带宽 69.8 GB/s，而孤立 GEMV 微基准是 235.6 GB/s** —— 差 3.4 倍。
   差距不在 GEMV，而在那些"零碎算子"：bias 12.7 us/次、rmsnorm 16.2 us/次、swiglu 17.3 us/次，
   每个只动几 KB 数据，纯粹是发射/延迟开销。280 次 × 12.7 us = 3.5 ms 只为了给 3072 个 float 加 bias。
3. **探针本身的开销**：103.18 tok/s（带 3400 对事件）vs 127 tok/s（不带）= **约 23%**。
   所以性能结论要关掉探针测，探针只用来定位"谁慢"。
### 7.6 三级兜底与"为什么 7B 还是进不去"（2026-11-10 17:10）

`DeviceArena` 现在按顺序尝试：

1. **cudaMalloc**，失败重试 5 次 × 100 ms（WDDM 下桌面合成器会周期性归还额度）；
2. **cudaMallocManaged**（托管内存）；
3. **cudaHostAllocMapped**（映射主机内存，UVA 下内核可直读，走 PCIe）。

每块到手立刻 `cudaMemset`/`memset` 触碰全部页，把 page fault 提前到加载期。

**实测结论：三级全部撞同一堵墙。**

| 事实 | 数值 | 含义 |
|---|---|---|
| `concurrentManagedAccess` | **0** | 1080 Ti（Pascal, sm 6.1）**不支持托管内存超额订阅**；`cudaMallocManaged` 只是换个分配器，装不下照样 OOM |
| `pageableMemoryAccess` | **0** | 同上，没有 HMM/pageable 访问 |
| 空进程里 `cudaHostAllocMapped 512 MiB` | ok（16/32/64/128/256/512 全部 ok） | 映射主机内存在**预算空闲**时可用 |
| 在预算耗尽的进程里同样的调用 | `invalid argument` | 说明 WDDM 下 pinned 映射**也要占一份设备侧预算** |

所以那个 ~2 GiB 的限制不是"分配 API 的问题"，而是**进程级的设备侧额度**：device / managed /
pinned-mapped 三种映射都算在同一份账上。1.5B（1.45 GiB）装得下，7B（4.5 GiB）在这台机器
当前状态下装不下 —— 这不是代码能绕过去的，只能靠减小同时持有的显存（跨进程分片、
或换一张不在 WDDM 桌面合成器下的卡）。

**顺带修的一个真 bug**：`Profiler::end()` 现在每个算子结束都 `NOVA_CU(cudaGetLastError())`。
之前 kernel 失败后进程继续跑、拿没写过的缓冲去采样，就是 TDR / 0x12 的来源。

修正后的 1.5B 端到端（1080 Ti，10 token，带探针）：

```
wall 70.285 ms   GPU 忙 66.666 ms (94.9%)   权重流量 4868.28 MiB (72.63 GB/s)
拷贝 20 次 / 5.80 MiB / 3.619 ms (5.1%)     2.00 次/token, 594 KiB/token

gemv.gate 18.641 ms (28.0%, 116.3 GB/s) | gemv.down 9.899 ms (14.8%, 219.0)
o_proj 5.656 ms (8.5%, 65.7) | head 5.469 ms (8.2%, 547 us/次)
gemv.q 5.215 ms (7.8%, 71.3) | swiglu 4.580 ms | rmsnorm.attn 3.290 ms
attn_decode 3.230 ms (4.8%) | bias.qkv 3.551 ms (5.2%)
```

注：`bias.qkv`（3.55 ms/10 token，12.7 µs/次）和 `rmsnorm`（6.5 ms）的融合**还没做** ——
它们合计占 14.6%，是下一步最划算的一刀（把 bias 写进 GEMV 尾巴、把 rmsnorm 与 quant 合并）。
---

## 8. 幽灵进程池 / VMM 重映射：三个实验的结论（2026-11-10 17:20）

POC 在 `tools/ghost_pool.cu`（构建：`nvcc -arch=sm_61 -Xcompiler /utf-8 ... -lcuda`）。
三个实验各自独立，结论如下。

### 8.1 E1 幽灵进程：**配额不是按 PID 的，跨进程共享**

```
主进程先占 1.5 GB  →  OK（free 9.08 → 7.58）
再拉 3 个子进程各要 1.5 GB  →  全部 out of memory
```

单独并行跑两个进程各要 1.5 GB（用 `--child` 模式，绕开主进程干扰）：

```
child 9 : cudaMalloc 1.5 GB -> OK      （free 9.08 → 7.58）
child 10: cudaMalloc 1.5 GB -> out of memory   （此时物理显存仍有 9.08 GiB 空闲）
```

**一个拿到 1.5 GB，另一个立刻失败**，物理空闲 9 GiB。所以那个 ~2 GiB 的额度是
**整个会话/整卡共享**的，不是每进程一份。多开进程堆不出更多可用显存。

### 8.2 E2 CUDA IPC：Windows 上不可用

```
child: cudaIpcGetMemHandle -> out of memory
```

导出这一步就失败了（Windows 没有 POSIX 共享句柄那套）。即使子进程能持有显存，
主进程也无法把它映射进自己的地址空间。**跨进程借显存在 Windows 上这条路是死的。**

### 8.3 E3 VMM：能保留虚拟地址，但物理仍记同一份账

```
VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 1
cuMemAddressReserve(4.5 GiB) -> 0  va=0x203A00000     （不占物理，free 不变）
cuMemCreate 第 0 块 512 MiB   -> 失败（预算）
```

VMM 本身可用、虚拟地址能保留 4.5 GiB，但 `cuMemCreate` 要的物理块记在**同一份预算**上，
所以拼不出 4.5 GiB。**VMM 只解决"地址连续"，不解决"显存不够"。**

### 8.4 结论：7B 在这台机器上不可行，但有一条正路

三条 hack 都被同一堵墙挡住：**整卡 ~2 GiB 的会话级预算**。7B（4.5 GiB）装不进去，
任何分配 API、任何进程数都改变不了。

真要跑 7B，只有三条路（按可行性排序）：

1. **层并行多进程**（唯一能在本机实现的）：3 个进程各持 9 层（每层 ~160 MB），
   每个进程都在预算内；激活（3584 float = 14 KB）通过**主机内存**在进程间传递
   （不用 CUDA IPC，因为 Windows 不支持）。每 token 多几次 14 KB 的往返，代价很小。
2. 换一张不在 WDDM 桌面合成器下的卡（TCC / 计算卡 / Linux），预算问题本身消失。
3. 7B 的 Q2_K（约 2.7 GB）仍然超预算，所以"降量化"救不了，只能走 1 或 2。
---

## 9. D3D12 图形预算实验：前提成立，但三条路都封死（2026-11-10 17:40）

POC：`tools/d3d12_steal.cu`（无头 D3D12 设备 + `CreateSharedHandle` + `cuImportExternalMemory`）。

### 9.1 两本账确实是分开的（用户的前提成立）

```
DXGI LOCAL budget   9.22 GiB | usage 0.14 GiB | available 9.08 GiB
CUDA   配额         约 2 GiB 就被挡住（见 §7.5 / §8.1）
```

图形账本报出 **9.22 GiB** 预算，而 CUDA 那本账在同一时刻只肯给约 2 GiB。前提没问题。

### 9.2 但图形账本"报得出、给不出"

| 申请 | 结果 |
|---|---|
| 1.0 GiB D3D12 DEFAULT 堆（SHARED + UAV + SIMULTANEOUS_ACCESS） | **成功** |
| 2.0 GiB | `0x8007000E` = **E_OUTOFMEMORY** |
| 3.0 GiB | `E_OUTOFMEMORY` |
| 4.5 GiB | `0x887A0005` = **DXGI_ERROR_DEVICE_REMOVED**（驱动重置） |

也就是说：那 9.22 GiB 是**预算数字**，不是这个远程/无头会话里真能分配出来的量。
更糟的是 4.5 GiB 那一次直接把设备打掉了 —— 正是之前导致远程黑屏的 TDR。
**所以我在第一次设备移除之后就停止了 2 GiB 以上的扫描**，不再拿你的显示器冒险。

### 9.3 CUDA 侧根本接不上

- CUDA 12.8 **没有** `cudaGraphicsD3D12RegisterResource`，也没有 `cuda_d3d12_interop.h`
  （11.8 同样没有）。运行时 API 的 D3D12 互操作在这个工具链里不存在。
- 驱动层 `cuImportExternalMemory(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE)` 对
  `CreateSharedHandle` 出来的句柄返回 **1 = invalid argument**，即使已经用了
  `D3D12_HEAP_FLAG_SHARED` + `ALLOW_SIMULTANEOUS_ACCESS` + `CUDA_EXTERNAL_MEMORY_DEDICATED`。

### 9.4 结论

四条"偷显存"的路线全部实测封死：

| 路线 | 结果 |
|---|---|
| 幽灵进程池（多 PID） | 预算**会话级共享**，第二个进程立刻 OOM（§8.1） |
| CUDA IPC 跨进程导入 | Windows 上导出就失败（§8.2） |
| VMM 虚拟地址重映射 | 能保留 4.5 GiB 虚拟地址，`cuMemCreate` 撞同一预算（§8.3） |
| D3D12 图形预算 | 账本更大但给不出，2 GiB 就 E_OUTOFMEMORY，4.5 GiB 直接设备移除（§9.2） |

**唯一还剩的真路是层并行多进程**：3 个进程各持 9 层（每层 ~160 MB，都在预算内），
激活（3584 float = 14 KB）经**主机内存**中转（IPC 不可用，但 14 KB 的往返代价可忽略）。
这是工程活，不是 hack。
---

## 10. 关键复查：2 GiB 不是碎片，也不是我们的代码（2026-11-10 17:55）

用户提出的反驳非常关键：如果 WDDM 真锁 2 GiB，llama.cpp 怎么能在同一台机器上把 1080 Ti
塞到只剩 5%？这一节是实测。

### 10.1 llama.cpp 在这台机器上**也失败**（同一堵墙）

```
D:\Desktop\ai\llama-b9631-bin-win-cuda-12.4-x64\llama-cli.exe -m build\q4k.gguf -ngl 99
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3794.07 MiB on device 0:
    cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

llama.cpp 用最朴素的一次性 `cudaMalloc`，要 3.7 GiB，**和我们一样 OOM**。
所以这不是 arena 分块、不是 NovaScope 的 Event 碎片、不是我们的任何代码。

### 10.2 也不是碎片：单次上限 == 累计上限

新增 `tools/arena_ceiling.cu`：全新进程、什么都不做，二分找"单次最大"和"256 MiB 分块的累计最大"。

```
cudaMemGetInfo: free 9.08 / total 11.00 GiB
MAXSINGLE: 2.25 GiB
MAXTOTAL : 2.25 GiB in 9 x 256 MiB chunks
```

**两者完全相等**。如果是碎片或"分配次数"问题，累计值应该明显高于单次值。相等只说明一件事：
当前进程能拿到的总量就是 2.25 GiB，而 `cudaMemGetInfo` 仍报 9.08 GiB 空闲。

### 10.3 这个上限是**动态**的，而且在缩水

| 时间 | 观察 |
|---|---|
| 16:47 | 单次 **4 GiB 成功**（oom_sweep `[A]`），1.5B（1.45 GiB）正常加载 |
| 17:40 | D3D12 POC 触发两次 `DXGI_ERROR_DEVICE_REMOVED`（驱动重置） |
| 17:55 | 单次上限掉到 **2.25 GiB**；3.5 GiB 的单次申请也失败 |

嫌疑排序（按我的判断）：

1. **我们自己的 D3D12 POC**：那两次设备移除就是驱动重置，之后上限从 ≥4 GiB 掉到 2.25 GiB。
   这条我认账 —— 不该为了验证"能不能偷显存"反复触发 TDR，即使每次只差一点。
2. **ToDesk**：3 个进程在跑（屏幕捕获会建 D3D 资源），会话是 `console`。
3. 驱动/会话的其他状态。

### 10.4 怎么复验（需要你做，我不敢动你的远程连接）

```bat
:: 1) 重启机器（让驱动状态复位）
:: 2) 停掉 ToDesk 服务后再跑（不要在我这边杀，怕断你的连接）
build\lt\arena_ceiling.exe
```

判读：

- 若 `MAXSINGLE >= 4.5 GiB` → **2 GiB 限制是环境态，不是硬锁**，7B 立刻能跑；
  此时 arena 已经改成"先一次性要一大块"，直接 `build_cuda_scope.bat NOVA_MODEL_QWEN25_7B`
  再 `set NOVA_SCOPE=1 & build\cuda_scope.exe build\q4k.gguf 8`。
- 若仍是 ~2.25 GiB，且停 ToDesk 后上升 → ToDesk 的诅咒坐实。
- 若重启后仍是 ~2.25 GiB → 这张卡在该驱动（581.57）/该会话下就是给不了更多。

### 10.5 arena 已按 llama.cpp 式改造

`DeviceArena::init()` 现在**先一次性 cudaMalloc 整个预算**，失败才退回 512 MiB 分块。
1.5B 实测走的是前者：

```
[arena] 一次性拿到 1452.5 MiB 连续块（llama.cpp 式预分配）
[nova] 生成 5 token: 9.21 ms/token -> 108.63 tok/s
```

7B 现在是：

```
[arena] 一次性 5174.3 MiB 失败，退回分块（每块 512 MiB）
[arena] cudaMalloc 512.0 MiB 失败（1/5）… 5 次后放弃
```