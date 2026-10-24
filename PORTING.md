> 仅对 V2.0.0-Core 生效。v2.0.0 之前的版本与 PR 不再支持。

# V2.0.0-Core 移植台账

## 0. 先说清楚

C:\Users\Administrator\Desktop\1080ti_kernal\ 里**只有 6 个 md，没有源码**。
第 1 节的算子清单是从文档反推的，不是从 conflux_nova.cu 读出来的。

## 1. 原工程内核清单（文档反推）

| # | 算子 | 出处 | 原接口形态 | V2 处置 |
|---|---|---|---|---|
| 1 | `quant_q8_1` | README `2.1 | warp 处理 128 元素超块 | → `launch_quant` |
| 2 | `conflux_mmq_load_tile_q8_0(_soa)` | prefill `1 | 显存→smem，x 按 k 块转置 | 布局保留；块格式改 Q4_K（未接） |
| 3 | `conflux_mmq_vec_dot_q8_0` | prefill `1 | int4 读 + 8×dp4a | → GEMV 内层同构 dp4a |
| 4 | `launch_conflux_gemv_soa_multi<NMAX>` | 批处理 `2 | out `[col*nrows+row]` | → `Kernel::q_proj<1>/<4>` |
| 5 | `launch_conflux_rms_norm_quant_multi` | 批处理 `2 | 逐槽位 x + 打包量化 | → `rms_norm` + `quant` |
| 6 | `launch_conflux_pack_quant4` | 批处理 `2 | N 行 fp32 → 打包 | → `launch_quant` |
| 7 | `launch_conflux_swiglu_quant_multi` | 批处理 `2 | silu*up 与量化融合 | → `launch_swiglu` |
| 8 | `launch_conflux_dequant_q8_0_to_f16` | prefill `5 | +3.07 GB fp16 副本 | 删 |
| 9 | `launch_conflux_gemm_f16` | prefill `5 | cuBLAS ALGO3 | 删 |
| 10 | `flash_attn` d64/d128/d128L | 自检输出 | 0.20 TMAC/s | → `flash_kernel<SWA>` |
| 11 | `attn_gemm_f32/chunk` | 自检输出 | batched GEMM，R=512 | 端口待做 |
| 12 | split-K + combine | README `2.7 | KV 切 32 段 | 端口待做 |
| 13 | `rope_kv` | 自检输出 | q/k 同转 | → `launch_rope`（布局编译期） |
| 14 | `qkv_post / sample_greedy` | 批处理 `2 | 逐槽位收尾 | → `residual` + `sample` |

## 2. 反转做了什么

**删掉（动态适配的全部痕迹）**：`KernelPlan` 函数指针表、`make_plan<Spec>()`、`kPlanOf`、
`plan_for(ModelId)` 五路 switch、`enum class ModelId`、运行期 `rows/cols` 参数。

**改成**：

| 新东西 | 位置 | 作用 |
|---|---|---|
| `-DNOVA_MODEL_*` → `using ActiveModel` | `model/select.h` | 一个二进制锁死一个模型，没选就 `#error` |
| `struct Kernel`（`q_proj / ffn_down / attn_decode ...`） | `kernel.h` | 直接点名张量 |
| `launch_gemv<K, R, NMAX>` | `src/kernel/q4k_gemv.cu` | K/R 是模板参数，越界即编译错误 |
| `ModelSignature + read_signature()` | `loader/gguf.h` | 只读文件，把特征抽出来 |
| `verify_model()` | `loader/validate.cpp` | 比对签名 / 告警 / abort |

## 3. 加载即校验（认签名，不认名字）

**第一步：读签名**（`read_signature`，纯读取，不做判断）

| 字段 | 来源键 |
|---|---|
| arch | `general.architecture` |
| layers | `{arch}.block_count` |
| n_embd | `{arch}.embedding_length` |
| n_head | `{arch}.attention.head_count` |
| n_head_kv | `{arch}.attention.head_count_kv` |
| head_dim | `{arch}.attention.key_length`，缺失则 `n_embd / n_head` |
| ffn | `{arch}.feed_forward_length` |
| vocab | `tokenizer.ggml.tokens` 的数组长度 |
| ctx | `{arch}.context_length` |
| rope_theta | `{arch}.rope.freq_base` |
| rms_eps | `{arch}.attention.layer_norm_rms_epsilon` |
| tied | `{arch}.tie_word_embeddings`（缺省 false） |

**第二步：比对**（11 个字段全比，不是"遇到第一个就停"）

- 全部一致 + 名字在白名单 → 放行；
- 全部一致 + 名字不在白名单 → 黄色 `[WARN]`，继续放行；
- 任一字段不同 → 打印差异表 → 精确 abort。

名字归一化再比：小写、剥掉 `-Q4_K_M / -q8_0 / -gguf / .gguf` 这类后缀，剩下的必须与
`kModelAliases` 精确相等。所以 `Qwen2.5-1.5B-Instruct-Q4_K_M` 是官方，
`Qwen2.5-1.5B-Uncensored` 是变种。

**第三步：张量表**——数量、名字集合、类型、形状逐项对；二维权重必须 Q4_K，norm 必须 F32，
Q6_K 直接拒（stock Q4_K_M 的 attn_v/ffn_down 就是 Q6_K）。
**第四步**：权重 ≤ 7 GiB，文件长度 ≥ data_offset + Σ bytes。

## 4. 实测

| 检查 | 结果 |
|---|---|
| CPU：Q4_K 反量化逐位 | 0 diffs |
| CPU：warp 分工点积 | rel 6.6e-08 |
| GPU：q_proj<1> | max_abs 1.8e-04 / rel_l2 1.5e-06 |
| GPU：q_proj<4> | max_abs 2.2e-04 / rel_l2 1.2e-06 |
| GPU：RoPE pos=0 恒等 | 0.0 |
| GPU：单键注意力 == V[0] | 0.0 |
| Loader 官方名（1.5B，338 张量） | 通过，0.809 GiB |
| Loader 变种名（同签名） | [WARN] + 通过 |
| Loader 7B 文件喂 1.5B 模具 | 差异表 5 行 + abort |
| Loader attn_v = Q6_K | abort |
| Loader 多一个张量 | abort |

## 5. 缺口

1. **权重上传**：现在只校验，没有把权重搬进 DeviceArena（`arena.h` 已就位）。
2. **逐层 forward**：KV 写入顺序、qkv bias、残差顺序还没写。
3. **prefill 性能路径**：`q4k_matmul` 是 4 列 GEMV 复用，MMQ 64×64 瓦片未接。
4. **decode split-K** 未接。
5. **Q4_K_M 的 Q6_K 张量**：模具只认纯 Q4_K，需要 `tools/requant`（未写）。

## 6. 待拍板

stock 的 `Qwen2.5-7B-Instruct-Q4_K_M.gguf` 里 `attn_v`/`ffn_down` 是 Q6_K，会在第三阶段被拒。

- **A（当前）**：离线 requant 成纯 Q4_K；
- **B**：模具接受 Q6_K，多一套 Q6_K 反量化内核 + 每槽位两个编译期实例。
