# DeepSeek-V2-Lite 在 nova_v2 上的移植：现状与设计

> 写于 2026-11-10。**代码先落地、数值先验**：小新（目标机）上 HF/ModelScope 都不通，
> 拿不到权重文件，所以先把不依赖文件的部分做完 —— 结果放在 `build_dsv2_host.bat` 一个命令里就能复现。
>
> 当前状态：**MLA（注意力）与 MoE 路由已实现并在 CPU 垫片上对拍通过**；
> 专家 FFN 调度、matrix-absorbed、27 层 runner 还没做（且被两件硬约束卡住，见 §5）。

---

## 1. 为什么 V2-Lite 不是「换个模型名」

`ModelSpec`（`include/nova/model/matrix.h`）能表达的是「dense FFN + 标准 GQA」。
V2-Lite 有两块结构上不存在的东西：

| 结构 | 现状 | 影响 |
|---|---|---|
| **MLA**（Multi-head Latent Attention） | 只有 `attn.cpp` 的 GQA：K/V 每 head 128 维，直接进 cache | 需要新的注意力 kernel（K 是 192 维 = nope 128 + rope 64），以及「kv_b 展开」这一步 |
| **MoE**（64 routed + 2 shared，top-6） | `spec.h: moe = false`、`dense.h: static_assert(!Spec::moe)` | 需要路由 + 按 idx 调专家 GEMV；模型层执行器要新写一份 |

这两条都不是「改个常量」。

---

## 2. 形状（`include/nova/model/deepseek_v2_lite.h`）

| 项 | 值 |
|---|---|
| 层数 / 隐藏 / heads | 27 / 2048 / 16（NKV 也是 16，MLA 展开后每 head 一套 K/V） |
| head_dim | 128（= v_head_dim = qk_nope） |
| MLA | kv_lora **512** + k_rope **64** = **576 / token / 层**（压缩态）；q/k 每 head **192** |
| 词表 / dense FFN | 102400 / 10944（只有第 0 层是 dense） |
| MoE | 64 routed，**top-6**，2 shared，expert FFN **1408** |
| rope | 只作用在每 head 的最后 64 维（Neox 前后半对），theta=10000 |

`arch/traits.h` 新增了 `DeepSeekV2`：`kRope = Neox`、`kQKVBias = false`、`kTiedEmb = false`。

---

## 3. 已实现并验证的部分

### 3.1 三个 kernel（`src/kernel/sycl/mla.cpp`）

| kernel | 做什么 | 布局 |
|---|---|---|
| `mla_rope_apply` | 只旋转 q/k 的 rope 部分（每 head 64 维） | 表复用 `rope_table`，**索引步长 2**（64 维 rope 的第 i 对 = 128 维表的第 2i 条） |
| `mla_kv_store` | 把展开后的 `k_nope`(128)、`k_rope`(64)、`v`(128) 写进 cache 第 pos 行 | `kc[ctx][NH][192]`、`vc[ctx][NH][128]` |
| `mla_decode` | 192 维 key、128 维 value 的 causal 注意力 | 一个 WG 管一个 head，4 个 subgroup 各扫一段 key，local 合并 `(m,l,acc)` |

**为什么 cache 存展开态而不是压缩态**：只存 576/token 当然更省（ctx=512 时 32 MB vs 283 MB），
但读的时候要对每个位置做 2 次 `512×4096` 的 GEMV，27 层 × 512 位置 ≈ 每 token 上千次 GEMV，直接不可行。
展开一次存下来，内层只剩点积。要省这块显存就上 matrix-absorbed（把 `kv_b` 折进 q/o），那是下一步。

### 3.2 MoE 前段（`src/kernel/sycl/moe.cpp`）

| kernel | 做什么 |
|---|---|
| `moe_router` | 一个 work-item 一个 expert，`sigmoid(W_router · x)`，64 个分数 |
| `moe_topk` | 单 work-item 选择排序取前 6（降序），按选中项之和归一化（`norm_topk_prob`） |

与 HF 的对应：`scoring_func='sigmoid'`、`topk_method='greedy'`、`norm_topk_prob=true`。
V3 的 group-limited routing 不涉及。

### 3.3 对拍结果（`build_dsv2_host.bat`，CPU 垫片跑真 kernel）

```
--- 1. mla_rope_apply（64 维 Neox rope）---      q max_abs 8.653e-08 | k max_abs 7.836e-08   PASS
--- 2. mla_kv_store（cache 布局）---             不一致 0 个；越界写 0 个                     PASS
--- 3. mla_decode（192/128 维 causal 注意力）---  n_keys=1: 0.0 | n=4: 6.7e-08 | n=8: 7.9e-08  PASS
--- 4. moe_router + moe_topk ---                 scores 3.2e-07 | idx 0 不一致 | w 4.5e-08 | sum 1.000000  PASS
```

参考实现全是 double；`n_keys=1` 时 rel_l2 恰好 0（softmax 退化成 1，输出必须逐位等于 v）。

---

## 4. 还没做的（按依赖顺序）

1. **专家 FFN 调度**：按 `idx` 对每个 expert 调 `gemv<>`（gate/up 形状 `[1408 × 2048]`、down `[2048 × 1408]`），
   加权累加 + 2 个 shared expert。这层是 host 侧驱动，kernel 复用现有 `gemv`。
2. **MLA 的 matrix-absorbed 变体**（省 283 MB cache、少 2 次 GEMV/层）。
3. **27 层 runner**：仿 `code/model/sycl/runner.cpp`，替换注意力与 FFN 两处。
4. **权重分页**：9.0 GB 权重 > 8 GiB 设备预算（`profile::kDeviceBudget`），要按层流式加载。

---

## 5. 两件硬约束（不解决就装不下 V2-Lite）

### 5.1 FFN / down 投影的行宽不是 256 的倍数 → Q4_K 表示不了

| 张量 | 行宽（ne[0]） | 256 整除 | 真实 GGUF 用的类型 |
|---|---:|---|---|
| `attn_q` | 2048 | ✓ | Q4_K |
| `attn_kv_a_mqa` | 2048 | ✓ | Q4_K |
| `attn_kv_b` | 512 | ✓ | Q4_K |
| `ffn_gate/up`（含 exps，行数为 10944/1408×64） | 2048 | ✓ | Q4_K |
| **`ffn_down`**（第 0 层） | **10944** | ✗（42.75） | **Q8_0** |
| **`ffn_down_exps`** | **1408** | ✗（5.5） | **Q8_0** |
| `ffn_down_shexp` | 2816 | ✓（11） | Q6_K |
| `output.weight` | 2048 | ✓ | Q6_K |

结论（不是推断，是从 GGUF 头里读出来的）：**上游不是靠 pad 解决，而是把 down 投影换成 Q8_0。**
所以引擎要跑 V2-Lite，缺的是 **Q8_0 与 Q6_K 的 GEMV**（两个 32 元素一块的格式，行宽只要 32 对齐），
而不是"pad 到 256"。这也意味着 `gemv<>` 不能只认 Q4_K，得做成量化类型可模板化。

### 5.2 权重 9.0 GB > 8 GiB 设备预算

Q4_K_M 整包 **9.65 GB**，超过 `profile::kDeviceBudget = 8 GiB`，单次分配上限 4 GB（已按 1 GiB 切块）。
真要跑，权重必须**按层流式**：每层 ~350 MB，算这一层前预取。这是 runner 里最重的一块工程量。

---

## 5.5 真实 GGUF 的逐张量布局（2026-11-10 核对）

来源：`mradermacher/DeepSeek-V2-Lite-GGUF` 的 `Q4_K_M`（9.65 GB，377 个张量，45 个 KV）。
只下了前 32 MB 的头就拿到了全部张量表 —— 这也是"没有完整文件也能做前期工作"的示范。

```
output.weight           Q6_K  ne=[2048, 102400]
token_embd.weight       Q4_K  ne=[2048, 102400]
blk.N.attn_norm.weight  F32   ne=[2048]
blk.N.attn_q.weight     Q4_K  ne=[2048, 3072]     = 16 head × (128 nope + 64 rope)
blk.N.attn_kv_a_mqa     Q4_K  ne=[2048, 576]      = kv_lora 512 + k_rope 64
blk.N.attn_kv_a_norm    F32   ne=[512]            ← 512 维隐向量上的 RMSNorm（原 spec 漏了）
blk.N.attn_kv_b         Q4_K  ne=[512, 4096]      = 16 head × (128 k_nope + 128 v)
blk.N.attn_output       Q4_K  ne=[2048, 2048]
blk.0.ffn_gate/up       Q4_K  ne=[2048, 10944]
blk.0.ffn_down          Q8_0  ne=[10944, 2048]
blk.N.ffn_gate_inp      F32   ne=[2048, 64]       ← 路由矩阵（fp32，64 个 expert）
blk.N.ffn_gate/up_exps  Q4_K  ne=[2048, 1408, 64]
blk.N.ffn_down_exps     Q8_0  ne=[1408, 2048, 64]
blk.N.ffn_gate/up_shexp Q4_K  ne=[2048, 2816]     ← 2 个 shared 融合成 2816
blk.N.ffn_down_shexp    Q6_K  ne=[2816, 2048]
```

三条对设计的修正（已写进 `deepseek_v2_lite.h` 的注释）：

1. **kv_a_norm**：`kv_a` 输出的前 512 维必须先做 RMSNorm，才能进 `kv_b`；
2. **shared expert 是融合的**：一个 2816 宽的 FFN，不是两个 1408；
3. **down 投影是 Q8_0、output 是 Q6_K**：见 §5.1。

KV 相关元数据：`expert_count=64`、`expert_used_count=6`、`expert_shared_count=2`、
`leading_dense_block_count=1`、`expert_weights_scale` —— 与 spec 完全一致。

---

## 6. 复现

```bat
build_dsv2_host.bat
build\sycl_dsv2_host_check.exe
```

只要 MSVC，不需要 GPU、不需要模型文件。**这条命令就是「V2-Lite 的算子对不对」的唯一答案。**

---

## 7. 与文件下载的关系

- 小新（192.168.137.8）改网络后：`huggingface.co` 的 DNS 时通时不通，HTTPS 仍然失败；
  `www.modelscope.cn` DNS 不通。**目前仍然拿不到 GGUF。**
- 本机：HF 不通，ModelScope 通，但 C: 只剩 0.4 GB、E: 有 21.6 GB。
- 也就是说：**下载这条路的瓶颈在网络，不在磁盘**；而代码这条路（本文档 §3）已经不受它影响。
