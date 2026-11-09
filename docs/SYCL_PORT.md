# SYCL 后端移植说明（Intel Arc 140T / Core Ultra 7 255H / Windows 11）

> 状态：**GEMV 已做方案 C（§11），其余算子仍是第一版**。本机没有 Intel GPU、没装 oneAPI，
> **因此这份代码没有用 `icpx` 编译过**。能在这台机器上证明的部分已经用两种方式验过：
> §9.1 用纯 C++ 复刻索引算术（旧映射），§9.2 用最小 SYCL 垫片把真 kernel 跑在 CPU 上
> （覆盖方案 C 的 SPLITS 1/2/4 × 连续/交错，`rel_l2` 全部 1e-7 量级）。
> 依赖 SYCL 运行时/驱动的部分（含全部性能数字）只能在目标机上验证，见 §8 / §11.5。
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
| `code/device/sycl/device.h` | 242 | 硬件 profile（写死常量）、`DeviceCaps`、USM 分配/上传/同步、`ChunkedWeight`（每块 ≤1 GiB）、`ActQ8View`/`Q4KWeightView` POD 视图、`NOVA_SYCL_SG`/`NOVA_SYCL_DP4A` 编译期开关 |
| `code/device/sycl/device.cpp` | 360 | 设备选择（优先 Arc）、in-order 队列、USM 记账、OOM 时打印 free/total 并 abort、分块切分与上传 |
| `code/device/sycl/dp4a_probe.cpp` | 44 | 编译期探针：只编译不运行，判断本机 oneAPI 的 `sycl::dp4a` 是哪种签名 |
| `src/kernel/sycl/sycl_common.h` | 262 | 模型形状（复用 `nova/model/matrix.h`）、调参宏、fp16 解包、dp4a 分派、16 B 装载、subgroup 归约、Q4_K lane 映射、**方案 C 的段切分** |
| `src/kernel/sycl/kernels.h` | 131 | 算子层唯一接口 + 显式实例化清单（含方案 C 的 A/B 变体） |
| `src/kernel/sycl/q4k_gemv.cpp` | 282 | **decode 主路径**：Q4_K × int8 GEMV（1 lane 持 16 B 权重）+ **方案 C 的 lane 重映射 + local 归约** + dp4a 微基准 |
| `src/kernel/sycl/q4k_gemv4.cpp` | 171 | 四列批量 GEMV：16 B 权重解包一次喂 4 组激活（仍是 1 subgroup = 1 行） |
| `src/kernel/sycl/quant.cpp` | 84 | fp32 → int8（每 32 一个 fp32 scale、每 16 一个 int 和） |
| `src/kernel/sycl/norm.cpp` | 110 | RMSNorm + 融合量化版 |
| `src/kernel/sycl/rope.cpp` | 155 | RoPE 表 + 应用（NeoX / 交错，编译期二选一） |
| `src/kernel/sycl/elem.cpp` | 219 | embed（Q4_K 取行）/ residual / bias / kv_store / swiglu+quant（融合） |
| `src/kernel/sycl/attn.cpp` | 260 | decode attention（split-K + combine）、prefill attention（tile，KV 进 local） |
| `tests/sycl_bench.cpp` | 923 | 基准：真张量、CPU 对拍、带宽、dp4a 加速比、tok/s 推算、其余算子抽检、**方案 C 的 A/B（§4c）** |
| `tests/sycl_mapping_check.cpp` | 128 | 不需要 GPU 的映射校验器（纯 C++ 复刻旧映射，本机已跑通，§9.1） |
| `tests/sycl_gemv_host_check.cpp` | 263 | **不需要 GPU 的 kernel 对拍**：用垫片把真 kernel 跑在 CPU 上，覆盖 SPLITS 1/2/4 × 连续/交错（§9.2） |
| `tests/host_sycl/sycl/sycl.hpp` | 410 | 最小 SYCL 垫片，只服务上面那个对拍，**不进 `build_sycl.bat`** |
| `build_sycl.bat` | 104 | 构建入口（探针 + icpx） |
| `build_sycl_host.bat` | 91 | **不需要 oneAPI** 的对拍构建入口（MSVC `cl` + 垫片），产出 `build\sycl_gemv_host_check_*.exe` |
| `docs/SYCL_PORT.md` | 本文件 | — |

合计约 3475 行（不含测试脚手架；另有 673 行 `tests/host_sycl` + `tests/sycl_gemv_host_check.cpp`）。

---

## 3. 算子映射

### 3.1 总表

| 算子 | CUDA 源 | SYCL 源 | 线程映射 | llama.cpp 参照（逐段对照过） |
|---|---|---|---|---|
| Q4_K GEMV | `src/kernel/q4k_gemv.cu : gemv_kernel` | `q4k_gemv.cpp : GemvQ4K<K,HW,SPLITS,STRIPE>` | **方案 C**：`SPLITS`(=4) 个 subgroup = 1 行，各扫一段连续 super-block；段内 lane 持 16 B、`kBlocksInFlight` 个块在飞；行内 4 个部分和走 local 合并（`SPLITS=1` = 旧映射） | `mmvq.cpp : mul_mat_vec_q<QK_K,QI4_K,block_q4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1>` + `vecdotq.hpp : vec_dot_q4_K_q8_1_impl_vmmq` |
| 四列 GEMV | `q4k_gemv.cu : gemv4_kernel` | `q4k_gemv4.cpp : Gemv4Q4K` | 同上（仍是 1 subgroup = 1 行，未做方案 C），权重解包一次喂 4 列 | `mmvq.cpp : mul_mat_vec_q_ncols<ncols_dst>`（第 304-366 行，read weight block once, dot against all columns） |
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

SYCL（`q4k_gemv.cpp`，一个 subgroup = 一行的一段（`SPLITS=1` 时就是整行），`HW`/`SPLITS`/`STRIPE` 是模板参数）：

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
| `NOVA_SYCL_GEMV_ROWS` | 8 | 每个 work-group 里有几个 subgroup（方案 C 下 8/`GEMV_SPLIT` = 2 行/WG）。调大 → 单 WG 更大、occupancy 降；上限被 static_assert 卡在 512 work-item |
| `NOVA_SYCL_GEMV_SPLIT` | 1 | 一行由几个 subgroup 分段扫（1 = 旧映射）。**默认已按实测回退到 1**：4 段实测更慢，见 §11.6 |
| `NOVA_SYCL_GEMV_STRIPE` | 0 | 1 = 4 个 subgroup 交错扫一行（段间不连续）。实验臂，只在 `sycl_bench` 的 A/B 里出现，默认不开 |
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

> **本节是移植前（2026-10）的预估，不是现状。** 目标机实测见 `docs/PERF_BASELINE.md`：
> GEMV 有效带宽只有 **18.27 GB/s**（§6.2 里"大概率过 70"的判断错了），差的不是访问宽度
> 而是访存形状，下一阶段是 §11 的方案 C。下面这张表留着，是为了让"当初怎么估的"可追溯。
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
:: 0) 不需要 GPU 的两步：先确认索引算术，再确认方案 C 的 kernel 体（本机都已跑过，§9）
::    cl /nologo /std:c++17 /O2 /EHsc /utf-8 /Iinclude tests\sycl_mapping_check.cpp src\loader\gguf.cpp /Fe:build\sycl_mapping_check.exe
::    build\sycl_mapping_check.exe build\q4k.gguf
::    cl /nologo /std:c++17 /O2 /EHsc /utf-8 /wd4068 /wd5030 /Iinclude /Icode\device\sycl /Isrc\kernel\sycl /Itests\host_sycl ^
::       /DNOVA_SYCL_MODEL_QWEN25_1P5B /DNOVA_SYCL_DP4A=0 /DNOVA_SYCL_SG=16 ^
::       tests\sycl_gemv_host_check.cpp src\kernel\sycl\quant.cpp src\loader\gguf.cpp /Fe:build\sycl_gemv_host_check.exe
::    build\sycl_gemv_host_check.exe build\q4k_15b.gguf

:: 1) 构建（首次 2-5 分钟；会先跑 dp4a 探针）
build_sycl.bat

:: 2) 主基准（必报五项 + 微基准 + 冷缓存 + 算子抽检 + 方案 C 的 A/B(§4c)）
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
| 2 | `rel_l2`（§3）+ §3b 的四个变体 | 全 < 1e-4（预期 1e-7 量级）；方案 C 的变体不许为了速度牺牲 |
| 2 | `有效带宽` | ≥ 70 GB/s；若 > 136.5 说明是 L3，看第 5 节冷缓存 |
| 2 | §4c 的 A/B 表 | 4 段/2 段比基线是否**显著**（差值 > 2σ）；落到 26~30 GB/s 才算方案 C 成立 |
| 2 | `dp4a 加速比` | 见 §7.2，单列 1.0-1.2 属正常 |
| 2 | `微基准 指令级加速比` | ≥ 2.5 说明 dp4a 是单条指令；约 1 说明被展开（oneAPI/驱动的问题） |
| 2 | `冷缓存 ... GB/s` | 这是 7B 真实推理的带宽 |
| 2 | `1.5B decode` | ≥ 55 tok/s |
| 2 | `退出码` | 0 = 全过 |

---

## 9. 本机（无 GPU）已经验证了什么

分两层：§9.1 是宿主 C++ **复刻**索引算术（旧映射），§9.2 是**直接跑 kernel 源码**（方案 C 全部变体）。

### 9.1 复刻版：索引算术（`tests/sycl_mapping_check.cpp`）

`tests/sycl_mapping_check.cpp` 把 `q4k_gemv.cpp` 的 lane 映射、块内偏移、`get_scale_min_k4`、
每 16 和索引、`dx*(dd*sc*dlo - dm*mc*sm)` 的代数重排**逐位复刻**到宿主 C++，读真实张量
（`build/q4k.gguf` 的 `blk.0.attn_q.weight`，3584×3584）与"反量化 + double 累加"对拍：

```
K=3584 R=3584 bytes=7225344
BPR=14 NP=4
row 0: kernel=0.15685207 ref=0.15685212 rel=3.14e-07
rel_l2 = 1.009e-07   max_abs = 1.789e-06   (3584 rows)
```

注意它复刻的是**旧映射（SPLITS=1）**，方案 C 的映射它不认识。改 lane 映射之后，这一层
不再有代表性——所以才加了 §9.2。

### 9.2 真 kernel：`tests/sycl_gemv_host_check.cpp` + `tests/host_sycl/sycl/sycl.hpp`

`tests/host_sycl` 是一个约 400 行的 SYCL 垫片：只实现 `q4k_gemv.cpp` / `quant.cpp` 用到的
那一小撮 API（`nd_range`/`nd_item`/`sub_group`/`reduce_over_group`/`local_accessor`/`vec`/
`handler`/`queue`）。work-item 是线程、`it.barrier()` 是条件变量、`local_accessor` 是每 WG
一块堆内存，并且每轮结束有一个 WG 屏障来保持 lockstep。

于是 `#include "q4k_gemv.cpp"` 之后，**生产线上的那份 kernel 体**（不是复刻）能在 CPU 上直接跑，
激活走生产代码 `quant_rows`，参考值是"反量化 + double 累加"。构建与运行：

```bat
build_sycl_host.bat 1.5b 16
build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf
build_sycl_host.bat 7b 16
build\sycl_gemv_host_check_7b_sg16.exe build\q4k.gguf
```

它现在覆盖四类东西：

| 检查 | 内容 | 判据 |
|---|---|---|
| §0 `quant_rows` | 逐元素对拍（int8 / 16-元素和 / scale） | 必须 **0 个不一致** |
| §0b `rmsnorm_quant` | y 对 double 参考 + q8 自洽（每块 32 个元素都被量化） | rel_l2 < 1e-6 且 0 个不一致 |
| §0c `swiglu_quant` | 同上 | rel_l2 < 1e-5 且 0 个不一致 |
| §1/§2 GEMV | 四个 lane 映射变体 + 尾部守卫 | rel_l2 < 1e-4 |

（2026-11-09 抓到的 SG=16 quant bug 就是 §0 这一行暴露的，见 `PERF_BASELINE.md` §10。）

结果（四个变体 × 两个张量 × SG 16/32 全 PASS，`rel_l2 < 1e-4`）：

| 张量 | SG | SPLITS=1 | SPLITS=2 | SPLITS=4 连续 | SPLITS=4 交错 |
|---|---:|---:|---:|---:|---:|
| 1.5B（BPR=6） | 16 | 9.706e-08 | 9.257e-08 | 9.505e-08 | 9.147e-08 |
| 1.5B（BPR=6） | 32 | 7.862e-08 | 8.339e-08 | 8.233e-08 | 7.886e-08 |
| 7B（BPR=14） | 16 | 1.006e-07 | 9.641e-08 | 9.382e-08 | 9.412e-08 |
| 7B（BPR=14） | 32 | 9.218e-08 | 9.264e-08 | 9.126e-08 | 8.998e-08 |

顺带把「尾部守卫」也验了：`rows` 不是「每 WG 行数」的整数倍时，最后一个 group 里越界的行不会被写。

**这一步证明的是**：lane 映射、段切分、跨 subgroup 归约、`local_accessor` + barrier 的用法，
在语义上都是对的（barrier 用错会直接死锁或读串行，垫片里会当场炸）。

**这一步证明不了**：`[[sycl::reqd_sub_group_size]]` 与 `NOVA_SYCL_SG=16` 的组合是否被 DPC++ 接受、
`local_accessor` 在 Windows 驱动上的实际行为、以及**任何性能**。剩下的这些只能在目标机上验：

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
| GEMV 预取 | 1 深（w0 = next） | min(4, NP) 深（`kPrefetchDepth`） | 用寄存器换 MLP；iGPU 的延迟比 1080Ti 更长。NP 是每 lane 的迭代数，段短时自动减浅 |

`kMaxBatch` 仍是 4（`include/nova/config.h`），所以 gemv4 就是批量 decode 的终点；
要吃到 §7.2 的 8 列收益，得先改 `kMaxBatch`，那是模型层的事，本后端不擅自改。

---

## 11. 下一阶段：GEMV Lane 重映射（方案 C）

> 基线数字见 `docs/PERF_BASELINE.md`。本节只改 GEMV 的 lane 映射，不动 attention，
> 也不改 §7.1 的三条既有方向；方案 C 是其中第 3 条的落地版。
>
> **状态（2026-11-09 更新）：方案 C 已在目标机测完并被数据否决 —— 旧映射最快，
> 4 段连续 -2.7%、2 段 -14.1%、4 段交错 -7.5%，全部显著（σ<1%，3 轮，§11.6）。
> 默认值已回退到 `NOVA_SYCL_GEMV_SPLIT=1`；代码和四个实验臂留着，可用 `-D` 复现。**
>
> 更要紧的是 §11.1 那个前提本身是测量假象：GEMV 不是 18.27 GB/s（峰值的 30%），
> 而是 **36.8 GB/s（60%）**。原因见 `PERF_BASELINE.md` §7：冷态/并发/短测量把数压低了 2 倍多。

### 11.1 18 GB/s 和 61 GB/s 差在哪

`tools/bench_ab.cpp` 的 512 MB 顺序扫描中位数是 **61.38 GB/s**。同一台机器上，
GEMV 在 ROWS=8、SG=16 下只有 **18.27 GB/s**，约为前者的 30%。两者的指令数都不是瓶颈，
差在**访存形状**。

先把两个被写错的数改对（按 GGUF 里的真实张量算，`kQK=256`、块 144 B）：

| 张量 | K | super-block/行 | 一行字节 |
|---|---:|---:|---:|
| 1.5B `blk.0.attn_q.weight`（基线那个） | 1536 | 6 | 864 B |
| 7B `blk.0.attn_q.weight` | 3584 | 14 | 2016 B |

所以「一行 1536 字节、24 个 super-block」是错的：1.5B 的一行只有 6 块 / 864 B。
更重要的是，**一个 subgroup 内部并不是 144 B 跨步**，代码里是

```
lane l（l = 8*b + c）  ->  wrow[b].qs + 16c，16 B
```

也就是 8 个 lane 拼一条连续的 **128 B**，b 再换下一个块（块与块之间隔 16 B 的
`d/dmin/scales` 头）。真正不连续的地方有两处：

1. 每 128 B 之后有 16 B 的块头空洞（11% 的字节读进来只用到 1/9）；
2. **同一个 work-group 的 8 个 subgroup 分散在 8 行上**，同一时刻摸到的是 8 个互不相邻的
   864 B / 2016 B 区间，每个区间只取其中 288 B（SG=16 时一次 2 块）。L2 看到的是 8 条短流，
   不是 1~2 条长流。

这与「单次访问宽度不够」无关：

- Xe 的最大 load 就是 128-bit（16 B），现在每 lane 已经用满；
- 现有代码已经是多深预取（旧映射下 1.5B 3 深 = 48 B、7B 4 深 = 64 B 在飞）；
- 再加预取深度只增加寄存器压力，不改变地址不连续这件事。


### 11.2 方案 C：lane 从「按行」改成「按块」

把一行切成 4 段，4 个 subgroup 各领一段：

| | 旧映射（SPLITS=1） | 方案 C（SPLITS=4） |
|---|---|---|
| 每行 subgroup 数 | 1 | 4 |
| 每个 subgroup 的范围 | 整行（1.5B 6 块 / 7B 14 块） | 连续 1/4 段（1.5B 1/2/1/2 块，7B 3/4/3/4 块） |
| 段间地址 | — | 首尾相接，不重不漏 |
| 一个 WG 负责几行 | 8 | 2 |
| 行内归约 | subgroup 内 `sg_sum` | 4 个 subgroup 各出部分和，走 local |

切法是 floor：第 s 段 = `[s*BPR/4, (s+1)*BPR/4)`，段长最多差 1（`sycl_common.h : q4k_seg_begin/end`，
kernel 与测试用的是同一份算术）。

代码形状（`src/kernel/sycl/q4k_gemv.cpp`，`GemvQ4K<K, HW, SPLITS, STRIPE>`）：

```cpp
const int sg_id = it.get_local_id(0);        // 第几个 subgroup
const int split = sg_id % SPLITS;            // 本 subgroup 负责第几段
const int r_in  = sg_id / SPLITS;            // 同一行有 SPLITS 个 subgroup
const int row   = it.get_group(0) * (kGemvRows / SPLITS) + r_in;
...
const int b0 = q4k_seg_begin(split, SPLITS, BPR);
const int hi = q4k_seg_end  (split, SPLITS, BPR);
// 段内不变：8 个 lane 拼一条 128 B，kBlocksInFlight 个块在飞
...
const float part = sg_sum<float>(sg, acc);
if (lane == 0) part_[sg_id] = part;
it.barrier(sycl::access::fence_space::local_space);
if (split == 0 && lane == 0) {               // 每行第一个 subgroup 的 lane 0 写出
    float total = 0.f;
    for (int s = 0; s < SPLITS; ++s) total += part_[r_in * SPLITS + s];   // 顺序固定
    out_[row] = total;
}
```

三条与实现相关的硬约束：

1. **越界行不能 `return`**。旧代码在 `row >= rows` 时直接返回；现在后面有 `it.barrier()`，
   必须所有 work-item 一起到，所以改成 `if (row < rows_ && hi > b0)` 只跳过载入。
2. **`static_assert(K/kQK >= SPLITS)`**。K=1536 只有 6 块，4 段里不会有空段；但如果有人把
   SPLITS 调到 8，1.5B 就会出现空段，编译期就挡住。
3. **SPLITS=1 保留成对照臂**，走的还是原来的「一行一个 subgroup」路径（`if constexpr`
   分支里连 barrier 都没有），A/B 的基线就是它。

`kSyclSgCompute=16` 不变，`kSyclSgAttn=32` 也不变。方案 C 只作用于计算类 kernel。
gemv4（四列批量）**没有改**：它仍是「1 subgroup = 1 行」，`sycl_bench` 里用
「gemv4 第 0 列 vs 单列 gemv(SPLITS=1)」逐位对拍来钉住这一点。


### 11.3 落地状态：代码在，数值在本机验完，带宽只能在目标机量

**已经做完的（本仓库，2026-11-09）**

| 项 | 位置 |
|---|---|
| lane 重映射 + local 归约 | `src/kernel/sycl/q4k_gemv.cpp : GemvQ4K<K,HW,SPLITS,STRIPE>` |
| 段切分算术（kernel 与测试共用） | `src/kernel/sycl/sycl_common.h : q4k_seg_begin / q4k_seg_end` |
| 旋钮 | `NOVA_SYCL_GEMV_SPLIT`（**默认 1**，实测 4 更慢已回退）、`NOVA_SYCL_GEMV_STRIPE`（默认 0） |
| A/B 变体（同进程可切） | `kernels.h : gemv_variant<K,HW,SPLITS,STRIPE>`，实例化 1 / 2 / 4 连续 / 4 交错 |
| 数值对拍（不需要 GPU） | `tests/sycl_gemv_host_check.cpp` + `tests/host_sycl/sycl/sycl.hpp` |
| A/B 计时 | `tests/sycl_bench.cpp` §4c |

`gemv<>()` 和 `gemv_variant<>()` 最终都调同一个 `gemv_launch_one<K,HW,SPLITS,STRIPE>`，
所以下面的对拍测的就是生产线上的那份 kernel 代码。

**数值：在 CPU 上把真 kernel 跑起来对拍（实测）**

本机没有 Intel GPU、也没装 oneAPI，连 `icpx` 都编不了。所以用 `tests/host_sycl` 的
最小 SYCL 垫片（线程 = work-item、条件变量 = barrier、`local_accessor` = 每 WG 一块堆内存）
把 `q4k_gemv.cpp` 的 kernel 体原样跑在 CPU 上，激活走的是生产代码 `quant.cpp`，
参考值是「反量化 + double 累加」。门槛 `rel_l2 < 1e-4`：

| 张量 | SG | SPLITS=1（基线） | SPLITS=2 | SPLITS=4 连续（方案 C） | SPLITS=4 交错 |
|---|---:|---:|---:|---:|---:|
| 1.5B（K=1536, BPR=6） | 16 | 9.706e-08 | 9.257e-08 | **9.505e-08** | 9.147e-08 |
| 1.5B（K=1536, BPR=6） | 32 | 7.862e-08 | 8.339e-08 | **8.233e-08** | 7.886e-08 |
| 7B（K=3584, BPR=14） | 16 | 1.006e-07 | 9.641e-08 | **9.382e-08** | 9.412e-08 |
| 7B（K=3584, BPR=14） | 32 | 9.218e-08 | 9.264e-08 | **9.126e-08** | 8.998e-08 |

四个变体全部 PASS，全部在 1e-7 量级，与基线的 8.437e-08 同级。两种映射之差（`max_abs`）
只有 9.5e-7 ~ 1.9e-6，就是「求和次序不同」的量级，不是映射错。
尾部守卫也验了：`rows = R-1`（不是 rows-per-WG 的整数倍）时最后一个 group 里越界的行
不会被写，前 R-1 行仍然全对。

**没验的**：带宽、tok/s、dispatch 占比。这三个必须在目标机（Arc 130T + oneAPI）上量，
见 §11.5 和 `docs/PERF_BASELINE.md` §6。

### 11.4 代价与风险：别指望线性的 1.4~1.6 倍

方案 C 拿掉的只是「一个 WG 摸 8 行」，但同时动了另一个变量，必须先讲清楚：

**① 每 lane 在飞的载入数（PD）掉了。** `PD = min(kPrefetchDepth, NP)`，NP 是每条 lane 的迭代数：

| 配置 | BPR | 段长 | NP | PD | 每 lane 在飞 | 每 WG 在飞 | WG 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B SPLITS=1 | 6 | 6 | 3 | 3 | 48 B | 6 KiB | 192 |
| 1.5B SPLITS=4 | 6 | 2 | 1 | **1** | **16 B** | 2 KiB | 768 |
| 7B SPLITS=1 | 14 | 14 | 7 | 4 | 64 B | 8 KiB | 448 |
| 7B SPLITS=4 | 14 | 4 | 2 | **2** | 32 B | 4 KiB | 1792 |

grid 大了 4 倍（WG 数 192→768），但**驻留**的 work-item 数是硬件上限（Arc 130T 约
112 EU × SIMD16 × 7 thread ≈ 1.2 万 lane），所以 1.5B 上真正的在飞字节是**降**的。
延迟隐藏变差会直接吃掉合并窗口的收益——这正是 18 GB/s 与 61 GB/s 之间那 70% 差距的另一半
（§11.1 指出的是「地址不连续」这一半）。

**② 1.5B 是这套方案最不友好的形状。** BPR=6 切成 4 段是 1/2/1/2：段长 1 的那些 subgroup
一半的 lane 组在空转（`sbi >= hi` 被跳过）。BPR=14 的 7B 切出来 3/4/3/4，浪费小得多。
`PERF_BASELINE.md` 里那个 18.27 GB/s 恰好是在 1.5B 上量的。

**③ 块头空洞还在。** 128 B 之后那 16 B 的 `d/dmin/scales` 照样要读，占 11% 的字节。
这条只有 §7.1 的「权重 SoA 重排」能收。

所以 `PERF_BASELINE.md` §4 那张表的目标区间（26~30 GB/s / 25 tok/s）是**上限估计**，不是承诺。
真要判断，只有一条路：目标机上把 `SPLITS=1 / 2 / 4连续 / 4交错` 四个臂一起量。
`sycl_bench` 的 §4c 已经把这件事做成同进程交替（A/B/A/B × 3 轮、比中位数、差值 < 2σ 判不显著），
避免「先跑完 A 再跑完 B」被热漂移吃掉差值（`PERF_BASELINE.md` §5.3 的规矩）。

SPLITS=2 那个臂是专门留的对照：7B 上它把 PD 保持在 4（不掉在飞字节），只把「一个 WG 摸几行」
从 8 改成 4。如果 4 段输了而 2 段持平或变快，主因就是 ①。

### 11.5 目标机上的验收流程

```bat
:: 0) 数值先过（不需要 GPU，只要 MSVC；本机已跑过，见 §11.3）
build_sycl_host.bat 1.5b 16
build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf

:: 1) 1.5B（基线那条：SG=16，注意 build_sycl.bat 默认是 32）
icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel ^
     -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=0 -DNOVA_SYCL_SG=16 ^
     -DNOVA_SYCL_GEMV_ROWS=8 -DNOVA_SYCL_GEMV_SPLIT=4 ^
     -Iinclude -Icode\device\sycl -Isrc\kernel\sycl ^
     code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv4.cpp ^
     src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp ^
     src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp tests\sycl_bench.cpp ^
     -o build\sycl\sycl_bench_c.exe

:: 2) 跑（4c 那一节就是 A/B；§3b 是四个变体的 rel_l2）
::    参数不是可选的，每个都在防一个已实测到的坑（PERF_BASELINE.md §7）：
::      --ab-iters 2000   一次测量 ≈100 ms，太短（10 次 = 0.5 ms）进不了稳态
::      --ab-cooldown 0   短测量 + 空闲会让 GPU 掉回 idle 时钟（实测 50 -> 100 us）
::      --warm-ms 2000    先烧 2 s 再计时；冷启动比提频态慢 22%
::      --no-cold         冷缓存那一节的方法在这台机器上测不出来（差值会变负）
::    另外：同一时刻只允许一个测量进程，编译也不要和测量并行（并发污染实测差 65%）
build\sycl\sycl_bench_c.exe --gguf build\q4k_15b.gguf --iters 300 --ab-iters 2000 --ab-cooldown 0 --warm-ms 2000 --no-cold
```

看五处，顺序不能换：

| 步 | 看什么 | 判据 |
|---|---|---|
| 1 | §3b 四行的 `rel_l2` | 全 < 1e-4（预期 1e-7 量级）；有一行爆了就停下，不要看带宽 |
| 2 | §4a 的提频曲线 | 批1 → 批8 应该在爬升；`提频预热` 之后才进入稳态 |
| 3 | §4c 的「4 连续」比基线 | 差值 > 2σ 且中位数更小 → 方案 C 在这台机器上成立 |
| 4 | §4c 的「每臂每次测量时长」 | 必须 ≥ 50 ms；出现 `[WARN] 单次测量 ... 太短` 就加大 `--ab-iters` |
| 5 | §4c 带宽列 + §4d 热漂移 | 26.5 GB/s 左右是稳态基线；|漂移| > 10% 时只认同一轮内的相对差 |

结论只有三种，对应三个动作（**已结案：实际结果是第三种，见 §11.6；下面留作决策记录**）：

- **4 段显著更快**：保持默认 `NOVA_SYCL_GEMV_SPLIT=4`，把 `PERF_BASELINE.md` §4 的目标换成实测数；
- **4 段与基线不显著**：说明合并窗口不是主因，回到 §7.1 的 ①（每 lane 32 B 载入）和 ②（SoA 重排）；
- **4 段显著更慢**：先试 `-DNOVA_SYCL_GEMV_SPLIT=2`（7B 上不掉 PD），再不行回 1，
  并把「PD 掉了多少」和实测一起记进 `PERF_BASELINE.md`——那是下一次判断的依据。

不能只看 GB/s。映射错的 kernel 往往也能跑出好看的带宽，因为它读的是错地址上的连续数据。
这次能提前把这条钉住，靠的是 §11.3 的 CPU 对拍：真的把 kernel 体跑起来，而不是复刻索引算术。

### 11.6 目标机实测结论：方案 C 输了，默认已回退

2026-11-09，Arc 130T / 1.5B / SG=16，四个臂在同一进程内交替，每臂每次 **2000 次发射**
（≈72 ms）、3 轮、轮间不空闲：

| 臂 | 中位数 | 带宽 | vs 基线 | σ | 判定 |
|---|---:|---:|---:|---:|---|
| **SPLITS=1（旧映射）** | **36.07 us** | **36.79 GB/s** | — | 0.9% | 基线 |
| SPLITS=2 连续段 | 41.99 us | 31.61 GB/s | **-14.1%** | 0.8% | 显著变慢 |
| SPLITS=4 连续段（方案 C） | 37.06 us | 35.81 GB/s | **-2.7%** | 0.5% | 显著变慢 |
| SPLITS=4 交错段 | 38.99 us | 34.04 GB/s | **-7.5%** | 0.1% | 显著变慢 |

降频检查：轮3/轮1 = -1.5% / -0.6% / +0.0% / -0.2% → 稳态，差值可信。四个臂 `rel_l2` 全 PASS。

后续又跑了三次（默认 `SPLIT=1` 的二进制、quant bug 修好之后各一次），排序完全一致：

| 运行 | 基线 | 4 段连续 | 交错 | 2 段 |
|---|---:|---:|---:|---:|
| A | 36.07 us / 36.79 GB/s | -2.7%（显著） | -7.5% | -14.1% |
| B | 35.70 us / 37.17 GB/s | -4.0%（不显著） | -9.7% | -14.0% |
| C | 35.10 us / 37.81 GB/s | -9.8%（显著） | -14.2% | -17.5% |
| D（quant 修复后） | 34.17 us / 38.84 GB/s | -8.8%（显著） | -15.0% | -17.1% |

**四次里基线都是最快的，方案 C 从来没有正收益。** 详见 `PERF_BASELINE.md` §8.1。

为什么输，和 §11.4 写下的代价完全对得上：1.5B 的 BPR=6 本该是「4 段 = 每段 6 块」的理想形状，
实际切出来是 1/2/1/2，段长 1 的 subgroup 有一半 lane 组空转，同时 `PD` 从 3 掉到 1。
合并窗口省下的那点延迟，不够补这两条。

**这条线到此为止。** 下一步如果还要追剩下的 40%（36.8 → 61.38），按 §7.1 的方向走：
每 lane 32 B 载入、权重 SoA 重排（把每 144 B 块里的 16 B 头单独抽出来）。
前提同样是**先用 §11.5 那套测量协议**（长测量、同轮交替、单进程），否则又会量到 18 GB/s 那种假象。



---

## 12. 端到端 decode runner（2026-11-09，第一次真模型跑通）

前面 §1-§11 都是**算子层**：每个 kernel 单独对拍、单独计时。这一节是**第一次把 28 层串起来**，
用真模型、真 KV、真采样跑出一次完整 decode。

### 12.1 在哪、怎么跑

| 文件 | 作用 |
|---|---|
| `code/model/sycl/runner.h` / `.cpp` | `sycls::E2ERunner`：加载 GGUF → embed → 28 层 × 12 算子 → head → 采样 |
| `tests/sycl_e2e.cpp` | 命令行驱动：分词 → 喂 prompt → 逐 token 生成 → 打印 tok/s 与文本 |
| `build_sycl_e2e.bat` | 构建入口（icpx + 全算子层 + runner） |

```bat
build_sycl_e2e.bat 1.5b
build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 24
```

### 12.2 与 CUDA 版的对应关系

| CUDA（code/graph/dense.h + src/model.cpp） | SYCL（runner.cpp） |
|---|---|
| AttnNormOp：rms_norm + quant | `rmsnorm_quant(h, A, x, attn_norm)`（融合，省一次 dispatch） |
| QkvProjOp：q/k/v 三个 gemm_q4k | 三个 `gemv<kD, true>` |
| QkvBiasOp | 三个 `bias_add`（`Model::qkv_bias` 编译期决定有没有） |
| RopeKvOp：rope_batch + kv_store_paged | `rope_apply` + `kv_store`（连续 KV，非 paged） |
| AttnOp：attn_decode_paged | `attn_decode` |
| OProjOp：quant + o_proj | `quant_rows<kD>` + `gemv<kD, true>` |
| ResidOp | `residual_add` |
| FfnNormOp / GateUpOp / SwiGluOp / DownOp | `rmsnorm_quant` / 两个 `gemv<kD, true>` / `swiglu_ffn` / `gemv<kFFN, true>` |
| head()：rms_norm + quant + lm_head | 同构；tie 时 lm_head 直接复用 token_embd |
| sample_last()：top-k 40 / top-p 0.95 / temp 0.8 | 同一套参数，host 侧（SYCL 还没有 sample kernel） |

**三处有意的简化**（里程碑切割，不是能力缺失）：

1. KV 是**连续布局** `[L][ctx][NKV*HD]`，不是 CUDA 的共享块池 + 块表 → 没有 paged / COW；
2. prompt 是**逐 token 喂**的（decode 口径），没有走 `attn_prefill` 的批量路径；
3. 采样在 host 上做（每 token 拉回 V 个 float = 608 KB）。

### 12.3 第一次跑通的实测（Arc 130T / 1.5B / SG=16 / DP4A=0）

```
[e2e] 模型: DeepSeek R1 Distill Qwen 1.5B Abliterated（151936 个 token）
[e2e] 模板: r1-abliterated        <- 按词表自动探测
[e2e] prompt 4 个 token: 151644 108386 151643 151645
[runner] 形状: L=28 D=1536 NH=12 NKV=2 HD=128 FFN=8960 V=151936 ctx=512 tied=1
[runner] 权重加载完成: 953.8 MiB，耗时 0.95 s
[e2e] prompt 处理 4 个 token，111.24 ms（36.0 tok/s）
[e2e] 生成 24 个 token：28.14 ms/token -> 35.54 tok/s（decode，不含 prompt）

<think>
</think>
你好！请告诉我有什么可以帮助你的？
```

- **35.54 tok/s 是实测的端到端 decode**（不是带宽模型推算）。28 层 × 14 次 dispatch ≈ 392 次/token，
  dispatch 只占 ~2 ms（5.6%）。
- 实测有效带宽 = 953.8 MiB / 28.14 ms ≈ **33.9 GB/s**，与算子基准的 36.8~38.8 同量级。
- 与 §9.2 的推算（41~44 tok/s）差 15~20%：① 每层还有 rmsnorm/rope/attn/swiglu 等非 GEMV 算子；
  ② 真模型权重是**按层顺序流一遍**（每层 ~26 MB），不像基准里那个 1.27 MiB 张量一直待在 L3。

### 12.4 跑通过程中抓到的两个 bug（都不在 kernel 层）

1. **特殊 token 被 BPE 拆开**：`<|im_start|>` 本该是 1 个 token（151644），tokenizer 却拆成
   `<` `|` `im` `_start` `|` `>` 6 个 → 模型收到字面文本，prompt 全错、输出是乱的。
   修法：`Tokenizer` 加「`<` 开头、`>` 结尾的词表项整体匹配」（`init_special` / `starts_special`，
   最长匹配优先）。**这个 bug 同时影响 CUDA 路径**（同一个 tokenizer、同一套 render）。
2. **目标机上的 `q4k_15b.gguf` 不是 Qwen2.5-1.5B**：`general.name` 是
   `DeepSeek R1 Distill Qwen 1.5B Abliterated`，词表里没有 `<|im_start|>`，用的是
   `<｜User｜>` / `<｜Assistant｜>`（全角 ｜）。`sycl_e2e` 现在按词表**自动选模板**
   （chatml / r1 / none），不再写死 ChatML。

### 12.5 下一步（按性价比）

1. **批量 prefill**：接 `attn_prefill`，prompt 一次前向（现在 23 token 的 prompt 要 23 次前向）；
2. **paged / COW KV**：多会话的前提，CUDA 版已有（`graph/kv_pool.h`）；
3. **sample kernel**：省掉每 token 608 KB 的 logits 回传；
4. SoA 权重重排（§9.3 ②，11% 收益）——runner 跑通之后才值得做。