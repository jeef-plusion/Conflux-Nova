# SYCL 后端移植说明（Intel Arc 140T / Core Ultra 7 255H / Windows 11）

> 状态：**第一版可交付代码**。本机没有 Intel GPU、没装 oneAPI，**因此这份代码没有在本机编译过**。
> 能在这台机器上证明的部分（Q4_K 的 lane 映射、索引算术、代数重排）已经用纯 C++ 复刻验证过，
> 见 §9；依赖 SYCL 运行时/驱动的部分只能在目标机上验证，见 §8。
> 不做能力探测、不做降级：profile 写死，分配失败直接 abort。

---

## 1. 两条命令出数据

```
build_sycl.bat
build\sycl\sycl_bench.exe --gguf build\q4k.gguf
```

第一条找 `icpx`（PATH 里没有就 `call oneAPI\setvars.bat`）、探 `sycl::dp4a` 的写法、
`-fsycl` 编出 `build\sycl\sycl_bench.exe`。第二条把要求的五项全部打印出来。
退出码 = 未通过的性能门槛数（0 = 全过）。

与 `build_core.bat`（CUDA sm_61 那条线）完全独立：不同的源文件、不同的输出目录、不共享任何宏。

---

## 2. 文件清单

| 文件 | 行数 | 职责 |
|---|---:|---|
| `code/device/sycl/device.h` | 235 | 硬件 profile（写死常量）、`DeviceCaps`、USM 分配/上传/同步、`ChunkedWeight`（每块 ≤1 GiB）、`ActQ8View`/`Q4KWeightView` POD 视图、`NOVA_SYCL_SG`/`NOVA_SYCL_DP4A` 编译期开关 |
| `code/device/sycl/device.cpp` | 360 | 设备选择（优先 Arc）、in-order 队列、USM 记账、OOM 时打印 free/total 并 abort、分块切分与上传 |
| `code/device/sycl/dp4a_probe.cpp` | 44 | 编译期探针：只编译不运行，判断本机 oneAPI 的 `sycl::dp4a` 是哪种签名 |
| `src/kernel/sycl/sycl_common.h` | 242 | 模型形状（复用 `nova/model/matrix.h`）、调参宏、fp16 解包、dp4a 分派、16 B 装载、subgroup 归约、Q4_K lane 映射 |
| `src/kernel/sycl/kernels.h` | 116 | 算子层唯一接口 + 显式实例化清单 |
| `src/kernel/sycl/q4k_gemv.cpp` | 207 | **decode 主路径**：Q4_K × int8 GEMV（1 lane 持 16 B 权重）+ dp4a 微基准 |
| `src/kernel/sycl/q4k_gemv4.cpp` | 170 | 四列批量 GEMV：16 B 权重解包一次喂 4 组激活 |
| `src/kernel/sycl/quant.cpp` | 84 | fp32 → int8（每 32 一个 fp32 scale、每 16 一个 int 和） |
| `src/kernel/sycl/norm.cpp` | 110 | RMSNorm + 融合量化版 |
| `src/kernel/sycl/rope.cpp` | 155 | RoPE 表 + 应用（NeoX / 交错，编译期二选一） |
| `src/kernel/sycl/elem.cpp` | 219 | embed（Q4_K 取行）/ residual / bias / kv_store / swiglu+quant（融合） |
| `src/kernel/sycl/attn.cpp` | 260 | decode attention（split-K + combine）、prefill attention（tile，KV 进 local） |
| `tests/sycl_bench.cpp` | 796 | 基准：真张量、CPU 对拍、带宽、dp4a 加速比、tok/s 推算、其余算子抽检 |
| `tests/sycl_mapping_check.cpp` | 128 | 不需要 GPU 的映射校验器（纯 C++，本机已跑通，§9） |
| `build_sycl.bat` | 104 | 构建入口（探针 + icpx） |
| `docs/SYCL_PORT.md` | 本文件 | — |

合计约 3230 行（不含注释约 2100 行）。

---

## 3. 算子映射

### 3.1 总表

| 算子 | CUDA 源 | SYCL 源 | 线程映射 | llama.cpp 参照（逐段对照过） |
|---|---|---|---|---|
| Q4_K GEMV | `src/kernel/q4k_gemv.cu : gemv_kernel` | `q4k_gemv.cpp : GemvQ4K` | 1 subgroup = 1 行；lane 持 16 B；4 个 lane 组并行抓 4 个 super-block | `mmvq.cpp : mul_mat_vec_q<QK_K,QI4_K,block_q4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1>` + `vecdotq.hpp : vec_dot_q4_K_q8_1_impl_vmmq` |
| 四列 GEMV | `q4k_gemv.cu : gemv4_kernel` | `q4k_gemv4.cpp : Gemv4Q4K` | 同上，权重解包一次喂 4 列 | `mmvq.cpp : mul_mat_vec_q_ncols<ncols_dst>`（第 304-366 行，read weight block once, dot against all columns） |
| fp32→int8 | `q4k_gemv.cu : quant_kernel` | `quant.cpp : QuantKernel` | 1 subgroup = 1 个 32 元素块（CUDA 是 1 thread 吃 32 元素） | `mmvq.cpp` 的 `quantize_q8_1` 路径；我们额外存"每 16 元素的和"（见 §3.3） |
| RMSNorm | `src/kernel/norm.cu : rms_norm_kernel` | `norm.cpp : RmsNormKernel` | 1 work-group = 1 行，两级归约（subgroup → local） | `norm.cpp : rms_norm_f32`（第 147-205 行） |
| RoPE | `src/kernel/rope.cu : rope_kernel` | `rope.cpp : RopeApplyKernel<NEOX>` | 1 work-item = 1 对元素 | `rope.cpp : rope_neox` / `rope_norm`（第 105-162 行） |
| embed | `src/kernel/elem.cu : embed_kernel` | `elem.cpp : EmbedKernel` | 1 WG = 1 token，256 个 item 覆盖一个 super-block | `getrows.cpp : get_rows_q4_K`（同构：Q4_K 取行后反量化） |
| residual / bias | `norm.cu : add_kernel` / `elem.cu : bias_kernel` | `elem.cpp` | 逐元素 | `element_wise.cpp` |
| swiglu | `norm.cu : swiglu_kernel + quant_kernel` | `elem.cpp : SwiGLUQuantKernel` | **融合成一个 kernel** | `element_wise.cpp : silu` + 我们的量化 |
| decode attn | `src/kernel/attn.cu : fa_split_kernel + fa_combine_kernel` | `attn.cpp : FaSplitKernel + FaCombineKernel` | 1 WG = 4 subgroup，各扫 1/4 段 KV，local 内合并 | `fattn-vec.hpp`（decode 的 split 结构）、`fattn-tile.hpp : flash_attn_tile`（local 合并写法） |
| prefill attn | `attn.cu : flash_kernel` | `attn.cpp : FaPrefillKernel` | 1 WG =（1 head，8 个 query），KV 按 16 key 一批进 local | `fattn-tile.hpp : flash_attn_tile`（tile + online softmax + local staging） |

### 3.2 GEMV 内层逐行对照

CUDA（`q4k_gemv.cu`，一行 = 一个 warp）：

```cuda
dlo = __dp4a(wv.x & 0x0F0F0F0F, al.x, dlo);
dhi = __dp4a((wv.x >> 4) & 0x0F0F0F0F, ah.x, dhi);
acc += dx.x * ((dd * scl) * (float)dlo - (dm * mcl) * (float)sm_lo);
```

SYCL（`q4k_gemv.cpp`，一行 = 一个 subgroup，`HW` 是模板参数）：

```cpp
dlo = d4<HW>(static_cast<std::int32_t>(static_cast<std::uint32_t>(w0) & kMask), al.x(), dlo);
dhi = d4<HW>(static_cast<std::int32_t>((static_cast<std::uint32_t>(w0) >> 4) & kMask), ah.x(), dhi);
acc += dx[0] * ((dd * (float)scl) * (float)dlo - (dm * (float)mcl) * (float)sm_lo);
```

三个必须注意的点（都写进代码了）：

1. **右移必须再掩一次**。CUDA 的 `>>` 作用在 int 上是算术右移，符号位会污染高 4 bit；
   这里统一写成 `(uint32_t)w >> 4 & 0x0F0F0F0F`，两种语义下都对。
2. **`w & 0x0F0F0F0F` 就是"解包"**。CUDA 的 gemv4 里写的是
   `(lo & 0xF) | ((lo>>8 & 0xF) << 8) | ...`，逐位等价于 `lo & 0x0F0F0F0F`（只是把每个字节的
   低 nibble 留在原地）。这里直接用掩码，少 12 条指令，数值完全相同。
3. **每 16 元素的 int 和（`xs`）替代 llama.cpp 的 dot2**。llama.cpp 为了算 Q4_K 的 -dmin*m
   那项，要再跑一次 `dp4a(0x01010101, u)` 把激活的字节和求出来；我们的 `ActQ8` 格式
   在量化时就把"每 16 个元素的和"算好存下来了（`quant.cpp` 里两次 `sg_sum`），
   kernel 里直接读一个 int，省 2 条 dp4a / 16 B 权重。

### 3.3 为什么 ActQ8 多存一个 int 和

`block_q4_K` 的 8 个子块各有独立的 (scale, min)。数学上
`out = sum dx[b] * (d*sc*q - dmin*m)`，第二项对 32 个元素是常数，可以提出来乘上"这 16 个
激活的和"。CUDA 版已经这么做了（`xs`），SYCL 版沿用同一布局，**两个后端的激活缓冲
逐字节可比**，方便跨后端对拍。

### 3.4 prefill attention 为什么必须重写

CUDA 版是"1 warp 1 个 (head,q)"，每个 q 独立把整段 KV 流一遍。KV 流量
= `n_q * NH * n_past * HD * 2 * 4B`，512 token 的 7B 模型约 7.5 GB —— 在 100 GB/s 的
iGPU 上就是 75 ms，不可接受。SYCL 版改成 tile：一个 work-group 管 (head, 8 个 query)，
KV 按 16 个 key 一批搬进 local（16 KiB），8 个 query 共享一次搬运，KV 流量 /8。
`kPrefillKC=16` 是被 32 KiB local 上限卡死的（`2 * 16 * 128 * 4 = 16 KiB`）。

---

## 4. 内存：USM + 分块

| 决策 | 实现 | 理由 |
|---|---|---|
| 分配方式 | `sycl::malloc_device`，**不用 buffer/accessor** | UMA 上 device USM 就是系统内存，省掉 accessor 的依赖跟踪开销；也因此在 init 时直接 abort 掉没有 `usm_device_allocations` 的设备 |
| 权重分块 | `ChunkedWeight`：按整行切，每块 ≤ 1 GiB（`profile::kMaxChunkBytes`） | Windows 驱动单 buffer 上限 1.0 GiB；按整行切保证任何一行都完整落在一块里，kernel 里没有跨块指针 |
| 单次分配上限 | `profile::kMaxAllocBytes = 4294901760`（3.999 GiB） | 驱动的硬限制；超过就 abort 并打印明细，不让驱动返回 nullptr |
| 上传 | 逐块 `queue.memcpy` + 一次 `wait` | 4 GiB 权重一次性搬完；不走 buffer 拷贝路径 |
| 记账 | 5 个 `MemTag`，live/peak/allocs | 超过 8 GiB 预算立即 abort |
| OOM | 捕获 `sycl::exception` / `std::bad_alloc`，打印"申请量 / 本进程已占用 / 设备 total / 系统 total+avail / 两个上限" | 要求"立即报错并打印 free/total" |
| 队列 | `in_order` 单队列 | 与 CUDA 的 stream 语义一一对应；算子之间不需要显式 event 依赖 |

`ChunkedWeight::upload()` 是**按块顺序** memcpy，块内行序与 GGUF 完全一致 —— 不重排、不转换，
所以 7B 的 3.99 GiB 权重上卡就是一次顺序搬 4 GiB。

---

## 5. 调参旋钮（全部 `-D` 可覆盖）

| 宏 | 默认 | 作用 / 怎么调 |
|---|---:|---|
| `NOVA_SYCL_SG` | 32 | subgroup 大小（16 / 32）。16 时一行仍由 8 lane 一组覆盖，`kBlocksInFlight` 自动变 2 |
| `NOVA_SYCL_DP4A` | 探针决定 | 0 = 纯标量，1/2 = `sycl::dp4a` 两种签名，3 = 手写乘加。**由 build_sycl.bat 注入，不要手改** |
| `NOVA_SYCL_GEMV_ROWS` | 8 | 每个 work-group 处理几行（= 几个 subgroup）。调大 → 单 WG 更大、occupancy 降；上限被 static_assert 卡在 512 work-item |
| `NOVA_SYCL_RMS_WG` | 256 | RMSNorm 每行的 work-group 大小（D=3584 时 256 → 14 轮） |
| `NOVA_SYCL_QUANT_SG` | 8 | 量化 kernel 每 WG 的 subgroup 数 |
| `NOVA_SYCL_ATTN_DEC_WG` | 128 | decode split-K 的 WG（128/32 = 4 个 subgroup；SG=16 时自动变 8 个，static_assert 卡 1..8） |
| `NOVA_SYCL_ATTN_SPLIT` | 32 | decode 最大 split 数（KV 段数） |
| `NOVA_SYCL_PREFILL_TQ` | 8 | prefill 一个 WG 覆盖几个 query |
| `NOVA_SYCL_PREFILL_KC` | 16 | prefill 每次 stage 多少个 key（受 32 KiB local 限制，见 static_assert） |
| `NOVA_SYCL_ROPE_NEOX` | 1 | 1 = Qwen2 系（前后半对），0 = Llama 系（相邻成对）。**换模型必须改这个** |

调法：直接在 `build_sycl.bat` 的 `icpx` 行上追加 `-DNOVA_SYCL_GEMV_ROWS=4` 之类。

---

## 6. 性能模型与预估

### 6.1 带宽模型

decode 是纯带宽题：每生成 1 token 要把全部权重读一遍。

```
tok/s = BW_eff / (W_bytes + KV_bytes_per_token)
KV_bytes_per_token = 2 * L * ctx * NKV * HD * 4B      （本引擎 KV 是 fp32）
```

权重字节（按 `nova/model/matrix.h` 的 `weight_bytes()`，编译期算死）：

| 模型 | 权重 | 每 token KV（ctx=512） | 合计 |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 828 MiB | 28 MiB | 856 MiB |
| Qwen2.5-7B | 3.99 GiB | 56 MiB | 4.15 GiB |

在 70 / 100 / 136.5 GB/s 三档下：

| 模型 | 70 GB/s | 100 GB/s | 136.5 GB/s |
|---|---:|---:|---:|
| 1.5B | **78 tok/s** | 112 | 152 |
| 7B | **16.1 tok/s** | 23.0 | 31.4 |

门槛"1.5B ≥ 55 tok/s"等价于 **BW_eff ≥ 49 GB/s**，即只要 GEMV 打到设计带宽的 49% 就过。

### 6.2 三个门槛逐条判断（发布前自己先判）

| 门槛 | 判断 | 依据 |
|---|---|---|
| GEMV 有效带宽 ≥ 70 GB/s | **大概率过**（但见下面 L3 说明） | 这个 kernel 每 lane 每次发 4×16 B、448 个 work-group 铺满，是 iGPU 上最经典的流式模式；同代 140V 实测 vector-add 93-107 GB/s，GEMV 通常能到 75-90% |
| dp4a / 标量 ≥ 1.8× | **单列 GEMV 基本不可能过** | 见 §7.2 的算术推导：Q4_K GEMV 每字节只有 0.5 次 MAC，两条路径都被带宽压死 |
| 1.5B ≥ 55 tok/s | **随带宽走，大概率过** | 需要 BW_eff ≥ 49 GB/s |

**L3 陷阱（必须写在前面）**：基准用的 `blk.0.attn_q.weight` 只有 7.2 MiB，而 Arrow Lake-H 的
L3 是 24 MiB —— 这个张量整个驻留 L3，测出来的"带宽"可能是 200-400 GB/s（远超 136.5 的 DRAM
理论值），**不能当作真实 decode 带宽**。所以基准里除了必报项，还加了第 5 节"冷缓存"：
每轮先 `memset` 64 MiB 把 L3 冲掉，再把 GEMV 从 flush 里差分出来。7B 真实推理看冷缓存那个数。

### 6.3 端到端折扣

tok/s 表没算：dispatch 开销、温降、桌面合成抢带宽。经验折扣 **0.6-0.8**
（1.5B 的 dispatch 占比更高，取 0.6；7B 取 0.75）。所以：

- 1.5B：78 × 0.6~0.8 = **47-62 tok/s**（贴着 55 的门槛，能不能过取决于 dispatch 数）
- 7B：16.1 × 0.6~0.8 = **9.7-12.9 tok/s**

---

## 7. 达不到怎么办（不粉饰）

### 7.1 带宽

三个方向，按性价比排：

1. **把一 lane 的载荷从 16 B 提到 32 B**（一次发两条 int4 载入，kBlocksInFlight 变 8）：
   单 subgroup 的在飞字节数翻倍，把 MLP 拉起来，直接改善延迟隐藏。
   代价：NP 从 4 掉到 2，尾部那一次 sbi >= BPR 的浪费从 1/4 变 1/2（K=3584 时 BPR=14）。
   要做成"K 能被 8 整除时才用 32 B"的编译期分支。
2. **权重 SoA 重排**：把每个 144 B 块里的 d/dmin/scales（16 B）单独抽成连续数组。
   现在每读 128 B qs 顺带读 16 B 头部，用不上却占了 11% 的带宽；重排后 GEMV 只读 qs
   （128 B/块）+ 一个紧凑的 scale 数组（被 32 个 lane 共享，实际只读 1 次）。
   理论回收约 10% 带宽。代价：上卡时要做一次 4 GB 的重写（一次性）。
3. **把并行抓取的 super-block 从 4 提到 8**：改成"2 个 subgroup 合作一行、各抓 7 个块"，
   把 NP 从 4 压到 1，彻底消掉尾部浪费。代价：归约要跨 subgroup（走 local），每行多一次 barrier。

### 7.2 dp4a 1.8× —— 这条门槛在单列 GEMV 上算术不可能

把数算给你看（Xe 上 32-bit 整数运算按 1024 lane × 2.25 GHz ≈ 2.3 T op/s 估）：

| 项 | 每 16 B 权重 | 一行（2016 B = 126 个单元） | 3584 行 |
|---|---:|---:|---:|
| dp4a 路径指令数 | 8 dp4a + 8 掩码 + 2 fma ≈ 21 | 2646 | 9.5 M |
| 标量路径指令数 | 32 mul + 32 add + 16 shift/and ≈ 80 | 10080 | 36 M |
| 折算时间（2.3 T op/s） | — | 1.2 us / 4.4 us | **4.1 us / 15.7 us** |
| 权重读取时间（100 GB/s） | — | — | **72 us** |

两条路径的**计算时间都比访存时间小一个量级**，所以无论 dp4a 多快，端到端都被带宽压住：
单列 GEMV 的加速比上界约 1.0，最乐观（标量计算完全藏进访存）也就 **1.2×**。

**要 1.8×，必须提高"每字节权重摊到的 MAC 数"**（N = 共享一次权重读的列数）：

| N | 标量计算 | dp4a 计算 | 访存 | 标量/kernel | dp4a/kernel | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.7 us | 4.1 us | 72 us | 72 | 72 | 1.0× |
| 4 | 63 | 16 | 72 | 72 | 72 | 1.0× |
| **8** | **126** | 33 | 72 | 126 | 72 | **1.75×** |
| 16 | 251 | 66 | 72 | 251 | 72 | **3.5×** |

结论：**dp4a 的收益要到 8 列以上才显形**。这台机器上 `kMaxBatch=4`，所以：

- 单列 GEMV（decode 主路径）：加速比 ≈ 1.0-1.2×，**按现在的口径就是过不了**；
- gemv4（4 列）：≈ 1.0-1.3×；
- 8 列（投机解码 / batch=8 / prefill 的 MMQ）才可能到 1.7-1.9×；
- 16 列：3× 以上。

如果 Xe 的 32-bit 整数乘只有 1/4 速率（部分 Xe 世代如此），标量那列要 ×4：
1 列就有约 1.4×，4 列约 2.6×。**这个数必须实测**，基准里的"微基准"（纯寄存器 8 条 dp4a vs
32 条乘加）就是干这个的，它把"dp4a 是不是一条指令"和"kernel 是不是带宽受限"两件事分开。

三个可执行的方向：

1. **把 dp4a 门槛的口径改成"8 列批量"**：把 gemv4 的列数改成模板参数并实例化 8/16，
   基准里同时报 N=1/4/8 的加速比曲线。这是把门槛放到它成立的区间，不是放水。
2. **走 XMX**：prefill 的 MMQ 用 `sycl::ext::oneapi::experimental::matrix::joint_matrix`
   （int8，M=8/N=16/K=32）。decode 用不上（GEMV 的 M=1），但 prefill 的 dp4a 会被整段替掉。
   前提：目标机暴露 `aspect::ext_oneapi_matrix`（140T 是否暴露未查证，需上机打印）。
3. **放弃 dp4a，改走"权重解包 + fp16 FMA"**：Xe2 的 fp16 吞吐约为 int8 的 1/2 但指令更简单
   （不用掩码/移位），在纯带宽受限的场景下与 dp4a 持平，且绕开 Intel 驱动 int8 的历史坑
   （`docs/BACKEND_INTEL_255H.md` §6 的 device-lost / 错编译记录）。这条只作为**稳定性兜底**，
   不作为性能路线。

### 7.3 dispatch 开销（1.5B 的真对手）

每 token 的 dispatch 数：现在约 **8 次/层**（rmsnorm+quant 融合、q/k/v 各 1、o 1、残差 1、
gate/up 1、swiglu+quant 融合 1、down 1），28 层约 224 次 + attention 的 2-3 次 × 28。
按 15 us/次提交（DPC++ 在 Windows 上的量级）就是 3.4 ms/token —— 比 1.5B 的 12.4 ms
访存时间小，但已经吃掉 20%。三个方向：

1. 把 residual + rmsnorm + quant 三合一（省 28 次）、qkv 三个投影合成一次 gemv3
   （省 56 次）—— 目标 <150 次/token。
2. 用 `sycl::ext::oneapi::experimental::command_graph` 把整张图录成一次 submit
   （oneAPI 2024.1+，Windows 上对 Level-Zero 后端有效）。这是最大的一刀。
3. 基准里已经用"差分法"（N 次 vs 2N 次）把 host 提交开销从 kernel 时间里分离出来，
   目标机上直接能看到 dispatch 到底占多少。

---

## 8. 目标机验证命令（按顺序）

```bat
:: 0) 不需要 GPU 的一步：先确认 Q4_K 的索引算术没问题（本机已跑过，1.0e-07）
::    cl /nologo /std:c++17 /O2 /EHsc /utf-8 /Iinclude tests\sycl_mapping_check.cpp src\loader\gguf.cpp /Fe:build\sycl_mapping_check.exe
::    build\sycl_mapping_check.exe build\q4k.gguf

:: 1) 构建（首次 2-5 分钟；会先跑 dp4a 探针）
build_sycl.bat

:: 2) 主基准（必报五项 + 微基准 + 冷缓存 + 算子抽检）
build\sycl\sycl_bench.exe --gguf build\q4k.gguf

:: 3) 长跑（把计时噪声压下去）
build\sycl\sycl_bench.exe --gguf build\q4k.gguf --iters 300

:: 4) 1.5B（换模型要重新编，形状是编译期常量）
build_sycl.bat 1.5b
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf --iters 300

:: 5) 持久负载（看温降；重复 5 次，对比第 1 次与第 5 次）
for /L %i in (1,1,5) do build\sycl\sycl_bench.exe --gguf build\q4k.gguf --iters 1000 --no-checks
```

频率/功耗另开一个窗口看（GPU-Z 传感器页，或任务管理器"性能 - GPU"的时钟曲线）。

每步看什么：

| 步 | 关键行 | 判据 |
|---|---|---|
| 1 | `dp4a 探针 1/2: 通过` | 两个都失败 → §7.2 的三条路 |
| 2 | `rel_l2` | < 1e-4（预期 1e-7 量级） |
| 2 | `有效带宽` | ≥ 70 GB/s；若 > 136.5 说明是 L3，看第 5 节冷缓存 |
| 2 | `dp4a 加速比` | 见 §7.2，单列 1.0-1.2 属正常 |
| 2 | `微基准 指令级加速比` | ≥ 2.5 说明 dp4a 是单条指令；约 1 说明被展开（oneAPI/驱动的问题） |
| 2 | `冷缓存 ... GB/s` | 这是 7B 真实推理的带宽 |
| 2 | `1.5B decode` | ≥ 55 tok/s |
| 2 | `退出码` | 0 = 全过 |

---

## 9. 本机（无 GPU）已经验证了什么

`tests/sycl_mapping_check.cpp` 把 `q4k_gemv.cpp` 的 lane 映射、块内偏移、`get_scale_min_k4`、
每 16 和索引、`dx*(dd*sc*dlo - dm*mc*sm)` 的代数重排**逐位复刻**到宿主 C++，读真实张量
（`build/q4k.gguf` 的 `blk.0.attn_q.weight`，3584×3584）与"反量化 + double 累加"对拍：

```
K=3584 R=3584 bytes=7225344
BPR=14 NP=4
row 0: kernel=0.15685207 ref=0.15685212 rel=3.14e-07
rel_l2 = 1.009e-07   max_abs = 1.789e-06   (3584 rows)
```

也就是说：**最容易写错的那一层（索引算术）已经证明是对的**；剩下没验证的只有 SYCL API 的
用法（`reqd_sub_group_size`、`reduce_over_group`、`local_accessor`、`parallel_for<Name>`）
和驱动行为。

没验证、也无法在本机验证的：

- 所有 kernel 能否编过（`sycl::dp4a` 的签名靠探针兜底，其余 API 都是 SYCL 2020 的基础设施）；
- `[[sycl::reqd_sub_group_size(32)]]` 与 `NOVA_SYCL_SG=16` 的组合是否被驱动接受；
- 实测带宽 / dp4a 指令级加速比 / 温降；
- Intel 驱动在 int8 dot 上的历史坑（device lost、区间错编译）—— 基准第 8 节的对拍就是用来
  在目标机上抓这个的：**如果 rel_l2 突然变大或 device lost，先怀疑驱动版本**，看
  `docs/BACKEND_INTEL_255H.md` §6 的版本区间。

---

## 10. 与 CUDA 版的差异（有意为之）

| 项 | CUDA 版 | SYCL 版 | 原因 |
|---|---|---|---|
| 量化 kernel 的并行粒度 | 1 thread 吃 32 元素 | 1 subgroup 吃 32 元素 | 32 个 lane 合成一条 128 B 合并访问 |
| SwiGLU | swiglu 后再 launch 一次 quant | 融合 | 省一次 dispatch，y 只写一次 |
| RMSNorm | 每次单独 launch | 额外提供 rmsnorm_quant 融合版 | 同上 |
| prefill attention | 1 warp / (head,q) | tile（8 query × 16 key） | KV 流量 /8 |
| 权重 | 一整块 | 分块（≤1 GiB） | Windows 驱动上限 |
| GEMV 预取 | 1 深（w0 = next） | 4 深（kPrefetchDepth） | 用寄存器换 MLP；iGPU 的延迟比 1080Ti 更长 |

`kMaxBatch` 仍是 4（`include/nova/config.h`），所以 gemv4 就是批量 decode 的终点；
要吃到 §7.2 的 8 列收益，得先改 `kMaxBatch`，那是模型层的事，本后端不擅自改。