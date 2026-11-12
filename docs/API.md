# API 参考 · 每个参数说清楚

> 目标：**第一次读这个项目的人，能在十分钟内知道"调哪个函数、每个参数填什么、什么时候会崩"**。
> 所有签名都从 `include/nova/**`、`code/device/sycl/device.h`、`src/kernel/sycl/kernels.h` 抄来，
> **不是凭记忆写的**。约束条件（static_assert / NOVA_REQUIRE）也一并列出 —— 这些是"填错就炸"的边界。

平台：Windows x64 ｜ 后端：SYCL（`kernels.h`）/ CUDA（`include/nova/kernel.h`）

---

## 0. 先看这张地图

```
你写代码的位置
    │
    ├─ Device          拿队列、分配显存、上传/下载        ← §1
    ├─ ChunkedWeight   装一块 Q4_K 权重                  ← §2
    ├─ ActQ8View       装一份 int8 激活                  ← §3
    ├─ 算 子            gemv / quant / rmsnorm / attn …   ← §4
    ├─ E2ERunner       整模型 decode（不想碰算子就用它）   ← §5
    ├─ KvPool          分页 KV（CUDA 线）                 ← §6
    └─ Scope           性能探针                           ← §7
```

**所有算子都是异步的**：函数返回时 kernel 只是"提交"了，不是"算完"。
要读结果必须 `Device::sync()`（或 `q.wait_and_throw()`）。

---

## 1. Device（设备与显存）

```cpp
Device();                       // 打开默认 SYCL 设备；没有设备会 NOVA_ABORT
~Device();                      // 释放所有还活着的分配（会打印泄漏报告）
class Device {
    void* alloc       (std::uint64_t bytes, MemTag tag, const char* what);
    void* alloc_shared(std::uint64_t bytes, MemTag tag, const char* what);   // USM shared，CPU 也能读
    void  release     (void* p, MemTag tag, std::uint64_t bytes);
    void  upload      (void* dst, const void* src, std::uint64_t bytes);     // 同步
    void  download    (void* dst, const void* src, std::uint64_t bytes);     // 同步
    void  zero        (void* dst, std::uint64_t bytes);
    void  sync();                       // 等队列里所有工作真正完成
    sycl::queue& q();                   // 拿队列（传给算子时就是它）
    void  print_memory_report() const;
};
```

| 参数 | 类型 | 含义 | 约束 |
|---|---|---|---|
| `bytes` | `uint64` | 要多少字节 | 超过 `kDeviceBudget`（默认 8 GiB）→ **abort**，不会返回 nullptr |
| `tag` | `MemTag` | 归属分类：`Weights` / `Kv` / `Act` / `Scratch` / `Io` | 只用于统计，填错不会崩但内存报表会误导 |
| `what` | `const char*` | 人类可读的名字（进日志） | 建议写成 `"layer3.attn_q"` 这种 |
| `p` | `void*` | `alloc` 返回的指针 | **必须原样传回**，`release` 不做指针运算 |

**为什么用 alloc 而不是 `sycl::malloc_device`**：统一记账 + 一个地方做预算检查 + 泄漏可见。
`alloc` 失败会直接 abort（宁可崩也不静默降级）。

```cpp
Device dev;
float* x = static_cast<float*>(dev.alloc(rows * cols * 4, MemTag::Act, "act.x"));
dev.upload(x, host_x.data(), rows * cols * 4);   // 同步，返回即可用
dev.sync();
dev.release(x, MemTag::Act, rows * cols * 4);
```

---

## 2. ChunkedWeight（一块 Q4_K 权重）

```cpp
struct ChunkedWeight {
    void create (Device& dev, std::int64_t rows, std::int64_t row_bytes, const char* what);
    void upload (Device& dev, const void* src);            // 整块上传（按 chunk 顺序）
    void upload_chunk(Device& dev, int i, const void* src);
    void destroy(Device& dev);
    std::int64_t rows()      const;
    std::int64_t row_bytes() const;
    int          n_chunks()  const;
    const std::uint8_t* base() const;   // 第 0 个 chunk 的基址（单 chunk 时 = 整块）
};
```

| 参数 | 含义 | 约束 |
|---|---|---|
| `rows` | 张量的行数（输出维度） | 要 ≥ 1 |
| `row_bytes` | 每行字节数 | **Q4_K 必须是 `cols/256*144`**，否则 `gemv<>()` 里 `NOVA_ABORT` |
| `src` | host 侧整块数据 | 长度必须 = `rows * row_bytes` |

**为什么要分 chunk**：为了在显存碎片化时（1080 Ti 的 WDDM 那种情况）把大张量拆成多块分配。
单块能装上时它就是一个 chunk，`base()` 可直接当整块用。

```cpp
ChunkedWeight w;
w.create(dev, /*rows=*/1536, /*row_bytes=*/q4k_row_bytes(1536), "l0.attn_q");
w.upload(dev, host_bytes.data());
// … 用完
w.destroy(dev);
```

---

## 3. ActQ8View（int8 激活）

```cpp
struct ActQ8View {
    std::int8_t*  qs = nullptr;   // rows * cols          —— 量化后的 int8
    float*        d  = nullptr;   // rows * (cols / 32)   —— 每 32 元素一个 fp32 scale
    std::int32_t* s  = nullptr;   // rows * (cols / 16)   —— 每 16 元素的 int32 和
};
```

| 字段 | 大小（元素） | 谁写 | 谁读 |
|---|---|---|---|
| `qs` | `rows * cols` | `quant_rows` / `rmsnorm_quant` / `swiglu_quant` | `gemv` |
| `d` | `rows * cols / 32` | 同上 | `gemv` |
| `s` | `rows * cols / 16` | 同上 | `gemv` |

**`s` 是什么、为什么要有它**：`s` 是每 16 个元素的 int8 之和。`gemv` 用它把
`min * Σa` 这一项变成一次减法，省掉两次 dp4a —— 这是相对 llama.cpp 的主要差别之一。

```cpp
std::uint64_t need = act_q8_bytes(rows, cols);      // 三者总和
ActQ8View a;
a.qs = static_cast<std::int8_t*>(dev.alloc(rows * cols,             MemTag::Act, "a.qs"));
a.d  = static_cast<float*>(      dev.alloc(rows * (cols / 32) * 4,  MemTag::Act, "a.d"));
a.s  = static_cast<std::int32_t*>(dev.alloc(rows * (cols / 16) * 4, MemTag::Act, "a.s"));
```

---

## 4. 算子

> 约定：所有算子 `void` 返回、**异步**、参数都是设备指针。`rows` 是"多少行激活"，
> `cols/K` 是"每行多宽"，两者必须与权重的形状一致。

### 4.1 `gemv<K, HW>` —— 权重 × 激活，decode 的主路径

```cpp
template <int K, bool HW>
void gemv(const ChunkedWeight& w, const ActQ8View& a, float* out, sycl::queue& q);
```

| 参数 | 含义 | 约束 / 填错的后果 |
|---|---|---|
| `K`（模板） | 每行列数 | 必须是 256 的倍数；只能用 `kD`(1536) 或 `kFFN`(8960)，其它值链接失败 |
| `HW`（模板） | `true` = 走硬件路径 | **当前 `NOVA_SYCL_DP4A=0`，true 等价于标量**（见 §8） |
| `w` | Q4_K 权重 | `w.row_bytes()` 必须 == `K/256*144`，否则 `NOVA_ABORT` |
| `a` | int8 激活 | 必须能容纳 `K` 列（`qs ≥ K`、`d ≥ K/32`、`s ≥ K/16`） |
| `out` | 输出 | 长度 = `w.rows()` 个 float；**只写不读** |
| `q` | 队列 | 任意；in_order 队列能与前后 kernel 保序 |

**做什么**：`out[r] = Σ_c w[r][c] * a[c]`，Q4_K 反量化与 int8 点积融合在一个 kernel 里。
**不做什么**：不加 bias（用 `bias_add`）、不加激活（用 `swiglu`）。

```cpp
gemv<kD, true>(w_qkv, act, qkv_out, q);   // qkv_out 有 w_qkv.rows() 个 float
```

### 4.2 `quant_rows<K>` —— float 行 → int8 行

```cpp
template <int K> void quant_rows(ActQ8View dst, const float* x, int rows, sycl::queue& q);
```

| 参数 | 含义 | 约束 |
|---|---|---|
| `K` | 每行宽度 | 256 的倍数 |
| `dst` | 输出激活 | `qs ≥ rows*K` 等（见 §3） |
| `x` | 输入 float | `rows*K` 个 |
| `rows` | 行数 | ≥ 1 |

**每个 32 元素子块一个 scale**（不是每行一个）—— 这就是那个"致命 quant bug"的位置：
SG=16 时如果只处理半个块，会造成 int8 与 scale 不一致（已修，`q8_block<SG>`）。

### 4.3 `rmsnorm_quant` —— RMSNorm + 量化（融合）

```cpp
void rmsnorm_quant(float* y, ActQ8View dst, const float* x, const float* w,
                   int rows, sycl::queue& q);
```

| 参数 | 含义 |
|---|---|
| `y` | 归一化后的 float 输出（`rows*K` 个），**会写** |
| `dst` | 同一份数据的 int8 版本（`gemv` 直接吃） |
| `x` | 输入 float |
| `w` | 每列一个的权重向量（长度 `K`，不是矩阵） |
| `rows` | 行数 |

**一次 kernel 完成三件事**：平方和归约 → 归一化 × 权重 → 量化。省掉两次全量读。

### 4.4 `swiglu_quant` / `swiglu` / `swiglu_ffn`

```cpp
template <int K> void swiglu_quant(ActQ8View dst, float* y, const float* gate,
                                   const float* up, int rows, sycl::queue& q);
```

`y = silu(gate) * up`，同时量化进 `dst`。
`swiglu` 固定 `K = kD`（FFN 中间层宽 `kD`），`swiglu_ffn` 固定 `K = kFFN`
（`down` 的输入宽 `kFFN`）—— **用错会静默算错，看形状选**。

### 4.5 元素级

```cpp
void residual_add(float* acc, const float* x, int n, sycl::queue& q);      // acc += x（n 个 float）
void bias_add(float* y, const float* b, int n, int cols, sycl::queue& q);  // y[i] += b[i % cols]
```

`bias_add` 的 `cols` 是"一行多宽"，`n` 是总元素数；`b` 只有 `cols` 个。

### 4.6 `embed_rows` —— 查表 + 反量化

```cpp
void embed_rows(float* x, const std::uint8_t* emb_q4k, const int* ids,
                int n, sycl::queue& q);
```

`x` 输出 `n * kD` 个 float；`emb_q4k` 是整个 embedding 矩阵（Q4_K）；`ids` 是 `n` 个 token id。

### 4.7 KV

```cpp
void kv_store(float* kc, float* vc, const float* k, const float* v,
              int n, std::int64_t pos0, sycl::queue& q);
```

把 `n` 个新位置（从 `pos0` 开始）的 k/v 写进 cache。cache 布局 `[ctx][NH][HD]`，
所以 `kc/vc` 的容量必须 ≥ `(pos0+n) * NH * HD`。

### 4.8 RoPE

```cpp
void rope_table(float* cs, float* sn, sycl::queue& q);                       // 预生成 cos/sin 表
void rope_apply(float* qv, float* kv, std::int64_t pos,
                const float* cs, const float* sn, sycl::queue& q);           // 单 token
void rope_apply_batch(float* qv, float* kv, int n, std::int64_t pos0,
                      const float* cs, const float* sn, sycl::queue& q);      // prefill
```

`qv/kv` 是**原地**旋转（会被改写）。`pos` 是绝对位置（不是行号）。

### 4.9 `attn_decode` —— 单 token 注意力

```cpp
void attn_decode(const float* qv, const float* kc, const float* vc, float* o,
                 int n_past, float* scratch, sycl::queue& q);
constexpr int attn_decode_scratch_floats();   // 需要多少 scratch（按 kNH/kAttnSplit 算）
```

| 参数 | 含义 | 注意 |
|---|---|---|
| `n_past` | KV 里**已包含当前 token** 的位置数 | 差 1 就会漏算或多算一个位置，**这是最容易填错的参数** |
| `scratch` | 分块归约用的临时区 | 必须 ≥ `attn_decode_scratch_floats()` 个 float，太小会越界写 |
| `o` | 输出，`kNH*kHD` 个 float | |

### 4.10 MoE（DeepSeek-V2-Lite）

```cpp
void moe_router(const float* x, const float* w_router, float* scores, sycl::queue& q);
void moe_topk  (const float* scores, int* idx, float* w, sycl::queue& q);
```

`scores` 有 64 个（64 个 routed expert 的 sigmoid 分数）；`moe_topk` 取前 `DSV2::kTopK`（=6），
`idx` 按分数降序，`w` 是归一化权重。专家自己的 gate/up/down 复用 `gemv<>`。

### 4.11 MLA（DeepSeek-V2-Lite）

```cpp
void mla_rope_apply(float* qv, float* kv, std::int64_t pos, const float* cs, const float* sn, sycl::queue& q);
void mla_kv_store (float* kc, float* vc, const float* k_nope, const float* k_rope,
                   const float* v, std::int64_t pos, sycl::queue& q);
void mla_decode   (const float* qv, const float* kc, const float* vc, float* o,
                   int n_past, sycl::queue& q);
```

每 head 192 维（nope 128 + rope 64），V 每 head 128 维；cache 是展开后的
`[ctx][NH][192]` / `[ctx][NH][128]`。**与 `attn_decode` 不通用**（形状不同）。

---

## 5. E2ERunner（整模型）

```cpp
class E2ERunner {
    E2ERunner(const std::string& gguf_path, int ctx_max = 512);  // 构造即加载全部权重
    int  step(int token_id, int pos, bool greedy = true);  // 吃一个 token，吐下一个
    void warmup(int n);                                    // 预热 n 步（不计时）
    int  ctx_max() const;
    int  layers()  const;                                  // 28
    std::int64_t weight_bytes() const;
    int  last_token() const;
};
```

| 参数 | 含义 | 约束 |
|---|---|---|
| `token_id` | 上一步的 token id | 必须在词表范围内（151936） |
| `pos` | 当前绝对位置 | `0 ≤ pos < ctx_max`；越界会写坏 KV |
| `greedy` | `true` = argmax，`false` = 按概率采样 | |

**`step` 内部做了 14 个算子 × 28 层**：rmsnorm_quant → qkv → rope → kv_store → attn →
o → residual → rmsnorm_quant → gate/up → swiglu → down → residual → head → 采样。
它是**唯一会把 logits 拉回 host 的地方**（每 token 608 KiB，约 2.5% 的墙钟）。

```cpp
E2ERunner r(g, 2048);
r.warmup(2);
int tok = r.step(prompt.back(), /*pos=*/prompt.size() - 1);
for (int i = 0; i < 32; ++i) tok = r.step(tok, prompt.size() + i);
```

---

## 6. KvPool（分页 KV，CUDA 线）

```cpp
class KvPool {
    void init(int n_slices, std::int64_t ctx);   // n_slices 个会话，每个 ctx 长
    KvSlice* acquire();                          // 拿一个空切片
    void release(KvSlice*);                      // 归还（会 erase，别存指针）
    int  alloc_block();
    void free_block(int b);
};
```

`KvSlice::table` 是"逻辑块 → 物理块"的映射（`int32`，长度 `nblocks`），`table_dev` 是它的设备副本。
**`acquire()` 返回的指针在 `release()` 之后失效**（底层是 `std::vector`）——
这条是审查时发现的 P1，V2.5 会改成 `deque`/索引。

---

## 7. Scope（性能探针）

```cpp
class Scope {
    static Scope& get();
    Rec& begin(const char* name, Phase ph, std::int64_t bytes);   // 开始记一条
    void finish(Rec& r);                                          // 结束（异步则只记 host 时刻）
    void resolve_all(sycl::queue& q);                             // 一次同步，补齐设备时间
    Summary summarize() const;
    void print_report() const;
    void write_trace(const char* path) const;                     // Chrome/Perfetto JSON
    void write_csv(const char* path) const;
};
#define NOVA_SCOPED(name, bytes, expr)   // 自动 begin/finish，最常用
```

| 参数 | 含义 | 约束 |
|---|---|---|
| `name` | 算子名（进 trace） | 建议用 `gemv.down` 这种 `类别.对象` |
| `ph` | `Phase::Compute` / `Copy` / `Host` / `Other` | `Compute` 计入 GPU 忙比，`Copy` 不计 |
| `bytes` | 所有权重/拷贝字节 | 填 0 表示不参与带宽统计 |

**两个必须知道的坑**（都写进了头注释）：

1. **不要混用时钟**：`command_end` 是 GPU 单调时钟，`steady_clock` 是 host 时钟；
   相减会得到 `-5e14` 这种时间戳。
2. **不要持有 `Rec&` 跨过 `begin()`**：`recs_` 是 `std::vector`，扩容会让引用失效
   （`OpGuard` 已改成存下标）。

`Scope` 关闭时（`NOVA_ENABLE_SCOPE=OFF`）所有调用是空转，零开销。

---

## 8. 常见错误对照表

| 症状 | 原因 | 怎么查 |
|---|---|---|
| `NOVA_ABORT: gemv<K=…>: 权重行字节 … != K/256*144` | `row_bytes` 与 `K` 不匹配 | 检查模型宏（`NOVA_SYCL_MODEL_*`）是否与 GGUF 一致 |
| `NOVA_ABORT: 超过设备预算` | 分配超过 `kDeviceBudget` | `print_memory_report()` 看谁占着 |
| 输出全是 0 / NaN | 忘了 `dev.sync()` 就读结果 | 加 `sync()` 再读 |
| 注意力结果偏一点 | `n_past` 差 1 | 确认 `n_past` 含当前 token |
| `'type_traits' file not found` | 只 call 了 `setvars`、没先 `vcvars64` | 见 `BUILD.md` §3 |
| `LNK1104: libmmd.lib` | 用 MSBuild 而不是 Ninja | `-G Ninja` |
| 跑分每次都不一样 | 冷启动 / 并发 / 省电模式 | 用 `--warm-ms 2000`，单进程，插电 |

---

## 9. 一页纸完整例子（单次 GEMV）

```cpp
#include "device.h"          // syclb::Device / ChunkedWeight / ActQ8View
#include "kernels.h"         // gemv / quant_rows
#include "nova/loader/gguf.h"

int main() {
    nova::GgufFile g("build/q4k_15b.gguf");
    syclb::Device dev;

    // 1) 权重：GgufFile 只解析元数据 + 张量表，数据要自己按 offset 读
    const auto* ti = g.find("blk.0.ffn_down.weight");   // ti->ne[0] = 列数（最快维）
    const int rows = static_cast<int>(ti->ne[1]);       // ne[1] = 行数
    std::vector<std::uint8_t> host_w(ti->bytes);
    // 你自己读文件：seek(g.data_offset() + ti->offset) 然后 read(host_w.data(), ti->bytes)
    syclb::ChunkedWeight w;
    w.create(dev, rows, q4k_row_bytes(1536), "ffn_down");
    w.upload(dev, host_w.data());

    // 2) 激活：host float → int8
    std::vector<float> x(1536, 0.01f);
    syclb::ActQ8View a;
    a.qs = (int8_t*) dev.alloc(1536,             syclb::MemTag::Act, "a.qs");
    a.d  = (float*)  dev.alloc(1536 / 32 * 4,    syclb::MemTag::Act, "a.d");
    a.s  = (int32_t*)dev.alloc(1536 / 16 * 4,    syclb::MemTag::Act, "a.s");
    float* xd = (float*) dev.alloc(1536 * 4, syclb::MemTag::Act, "x");
    dev.upload(xd, x.data(), 1536 * 4);

    // 3) 量化 + GEMV（都是异步）
    quant_rows<kD>(a, xd, /*rows=*/1, dev.q());
    float* out = (float*) dev.alloc(w.rows() * 4, syclb::MemTag::Io, "out");
    gemv<kD, true>(w, a, out, dev.q());

    // 4) 等结果
    dev.sync();
    std::vector<float> y(w.rows());
    dev.download(y.data(), out, w.rows() * 4);
    // y 就是结果
}
```

---

## 10. 这份文档没覆盖的

- CUDA 线的算子签名（`include/nova/kernel.h`）：与 SYCL 版**语义相同、参数名不同**，
  以头文件为准；
- `dense.h` 的 `LayerCtx`：它是内部结构体（16 个字段的大杂烩），审查已标记为 P1，
  V2.5 会拆开 —— **不要拿它当公开接口**；
- 训练/微调：本项目只做推理，没有反向传播。

> 发现文档与代码不一致：**信代码**，然后提一个 PR 改这里（`CONTRIBUTING.md` §1 要求
> 附"功能特点/核心参数/逻辑流程"）。
