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
