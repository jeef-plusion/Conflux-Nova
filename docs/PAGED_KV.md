# Paged KV：共享块池与非连续寻址

> FJEC 2.2.0 核心重构设计稿。目标：跨会话复用 KV，零拷贝，支持 COW。

## 0. 三个前提

1. **块是共享的**：一个对话前缀被多个会话引用时，物理 KV 只有一份。
2. **共享是只读的**：任何写入都必须先确认独占，否则 COW。
3. **寻址是非连续的**：逻辑位置 → (块号, 槽位)，由每会话的 block table 决定。

## 1. 数据结构

```cpp
constexpr int KB = 16;                    // 每块 16 个 token
constexpr int KVW = 2 * Model::NKV * Model::HD;   // 每 token 的 float 数

struct KvPool {
    float*   base;        // [layer][block][slot][2][NKV][HD]
    int32_t* refcnt;      // 每块的引用计数（宿主侧）
    int      n_blocks;
    int      n_free;
    std::vector<int32_t> free_list;
};

struct SessionKV {
    std::vector<int32_t> table;   // 逻辑块 -> 物理块
    int64_t n_past = 0;           // 已写入的 token 数
    int     tail_ref = 1;         // 尾块引用计数快照
};
```

### 尺寸（我们的真实形状）

| 模型 | 每 token 每层 | 每块每层 (KB=16) | 每块全层 | 512 块池 |
|---|---|---|---|---|
| 7B (L28, NKV4, HD128) | 4,096 B | 64 KB | 1.75 MB | **896 MB** |
| 1.5B (L28, NKV2, HD128) | 2,048 B | 32 KB | 896 KB | **448 MB** |

对比现状：4 路 × 2048 ctx 独占切片 = 875 MB。**块池的内存占用同一量级，但可复用。**

block table 大小：2048 / 16 = **128 个 int32 = 512 B**，可忽略。

## 2. 映射公式

```
物理地址 = pool_base
         + (layer * n_blocks + table[p >> 4]) * (KB * KVW)
         + (p & 15) * KVW
```

`p >> 4` 与 `p & 15` 是**唯一**的额外运算。KV 是 fp32 且 HD=128，每 token 的 k 与 v 各 512 B（7B），天然对齐，块内仍是连续访问。

## 2.5 块大小由用户指定

`KB` 不是硬编码，而是**编译期可指定**：

```bat
build_core.bat NOVA_MODEL_QWEN7B                       :: 默认 16
nvcc ... -DNOVA_KV_BLOCK=32 ...                        :: 改成 32
```

允许 `8 / 16 / 32 / 64`，其余值直接编译期 `static_assert` 拦下。

**为什么是编译期而不是运行期**：块号与槽位靠 `pos >> shift` 与 `pos & mask` 算，移位量必须是常量。
否则每写一个 token 都要多一次分支或一次运行期除法——那正是我们整套架构要消灭的东西。

| 块大小 | 块表长度 (ctx 2048) | 7B 每块全层 | 尾块平均浪费 | 适用 |
|---|---|---|---|---|
| 8 | 256 | 896 KB | 4 token | 碎片敏感、短对话 |
| **16** | **128** | **1.75 MB** | **8 token** | **默认，均衡** |
| 32 | 64 | 3.5 MB | 16 token | 长上下文、跳转敏感 |
| 64 | 32 | 7 MB | 32 token | 只看吞吐、不在乎碎片 |

配置入口：`code/graph/kv_pool_cfg.h`（`kKvBlock` / `kKvBlockShift` / `kKvBlockMask`）。

## 3. 追加与 COW

会话是顺序追加的，所以**只有尾块可能被共享**。规则三条：

```
append(tok):
  p = n_past
  if p % KB == 0:                    // 起始新块
      blk = alloc_block(); refcnt[blk] = 1
      table.push_back(blk)
  else:                              // 落在尾块
      blk = table.back()
      if refcnt[blk] > 1:            // ← 共享且要写：COW
          nb = alloc_block()
          memcpy(pool[nb], pool[blk], KB*KVW*4)
          refcnt[blk]--; refcnt[nb] = 1
          table.back() = nb; blk = nb
  write_kv(pool, blk, p & 15, tok)
  n_past++
```

**关键性质**：前缀共享是只追加的，所以 COW 只在“尾块被共享且要继续写”时触发，每次最多拷一个块（7B 下 1.75 MB，且只发生一次）。

挂载共享块（命中前缀）：

```
attach(session, donor, up_to_block k):
  for b in 0..k: table.push_back(donor.table[b]); refcnt[donor.table[b]]++
  n_past = (k+1) * KB
  // 尾块若只填了一半，下次 append 会自动 COW
```

## 4. Paged decode attention（核心内核）

```
// grid: (n_heads_total, 1, n_sessions)   每 block = 1 个 (session, head)
__global__ void attn_decode_paged(
    const float* q,            // [S][NH][HD]
    const float* kv_pool,      // [layer][block][slot][2][NKV][HD]
    const int*   block_table,  // [S][max_blocks]
    const int*   n_keys,       // [S]
    float*       out)          // [S][NH][HD]
{
    const int s    = blockIdx.z;
    const int h    = blockIdx.x;
    const int lane = threadIdx.x;           // HD = 128 线程
    const int kvh  = h / (NH / NKV);
    const int P    = n_keys[s];             // 本会话 key 总数

    float m = -INF, l = 0, acc = 0;
    const float qv = q[(s*NH + h)*HD + lane];

    // 外层按块走，内层块内连续——保住合并访问
    for (int b = 0; b < (P + KB - 1) / KB; ++b) {
        const int phys = block_table[s*MAXB + b];
        const float* Kb = kv_pool + ((layer*NB + phys) * KB * KVW);
        const int lim = min(KB, P - b*KB);
        for (int t = 0; t < lim; ++t) {
            const float* kp = Kb + t*KVW + kvh*HD;
            float sc = qv * kp[lane];
            for (int o = 16; o; o >>= 1) sc += __shfl_xor_sync(0xffffffff, sc, o);
            const float m2 = fmaxf(m, sc);
            const float e  = __expf(sc - m2);
            const float c  = __expf(m - m2);
            l = l*c + e;
            acc = acc*c + e * (Kb + t*KVW + NKV*HD + kvh*HD)[lane];
            m = m2;
        }
    }
    out[(s*NH + h)*HD + lane] = acc / l;
}
```

要点：

1. **块号只在外层取一次**，内层 16 个 slot 连续 → 与现在的连续扫描几乎同带宽。
2. `block_table` 常驻常量内存或用 `__ldg`，每块一次。
3. 每块尾部的 `lim` 处理半满块。
4. 四路批量时 `blockIdx.z` 已经就是会话维度，无需改造成本。

## 5. kv_store 的对应改动

```
// 原：kv[l][pos]
// 新：(layer, 块号, 槽位)
int blk = table[p >> 4], slot = p & 15;
float* dst = pool + ((layer*NB + blk)*KB + slot)*KVW;
dst[kvh*HD + lane] = k_val;
dst[NKV*HD + kvh*HD + lane] = v_val;
```

prefill 写 N 个 token 时可能跨多个块，按 `p` 逐 token 计算即可（每 token 一次除法，可忽略）。

## 6. 落地顺序（三步，每步都可独立验证）

| 步 | 内容 | 验证 |
|---|---|---|
| 1 | 块池 + 映射，但**每会话独占**（refcnt 恒为 1，无共享） | 输出与现行逐字节相同 |
| 2 | paged attention 内核替换连续版 | 同上，且带宽掉幅 <5% |
| 3 | 共享挂载 + COW | 两个会话共享前缀，各自追加 10 轮互不污染 |

第 1、2 步不做共享，所以不会引入语义风险；第 3 步才是真功能。

## 7. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 非连续导致带宽下降 | 每块一次跳转，块内连续 | 块取 16 或 32 可调，实测后定 |
| 块内碎片 | 尾块平均浪费 8 个 token | 可接受（128 块表 × 16 = 2048 上限内） |
| COW 拷贝成本 | 每块 1.75 MB（7B） | 只发生在尾块，且仅共享时 |
| 引用计数竞争 | 多线程会话 | 现在单线程，加锁成本为 0 |
