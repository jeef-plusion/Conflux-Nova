# 设备后端贡献指南

> 这份文档写给想给 Conflux Nova 贡献新硬件后端的人。
> 一句话：**我们不规定性能，只规定存在性。**

## 为什么有门槛

计算图（模型适配层）是按算子拼出来的：

```
AttnNorm → QkvProj → [QkvBias] → RopeKv → Attn → OProj → Resid
         → FfnNorm → GateUp → SwiGlu → Down → Resid
```

少任何一个单元，`code/graph/` 就得为这个后端开特例——那正是我们要消灭的东西。**缺一件，这条线就破了。**

## 必须实现的单元（软约束，但不可缺）

| # | 单元 | 为什么不能省 | 参考实现（CUDA） |
|---|---|---|---|
| 1 | **矩阵乘**（GEMV / 批量） | 全部权重都在这里，占单步约 70% | `src/kernel/q4k_gemv.cu` |
| 2 | **softmax** | attention 的必要组成 | `src/kernel/attn.cu` 内 |
| 3 | RMSNorm | 每层两次，缺了数值直接崩 | `src/kernel/norm.cu` |
| 4 | RoPE | 位置信息，缺了长上下文失效 | `src/kernel/rope.cu` |
| 5 | 量化 / 反量化 | 权重是 Q4_K，不解包走不动 | `src/kernel/q4k_gemv.cu` + `quant` |
| 6 | elementwise | residual / swiglu / bias | `src/kernel/elem.cu` |
| 7 | embed / 采样 | 进出的两端 | `elem.cu` / `model.cpp` |

第 1、2 条是**最低中的最低**：没有矩阵乘，模型跑不了；没有 softmax，attention 不成立。

性能可以先很低（softmax 用标量循环写都行），但不能缺件。

## 提交时附带的三样东西

### 1. `BACKEND.md`

逐条说明上表七类：

| 单元 | 实现了吗 | 走硬件还是软件模拟 | 备注 |
|---|---|---|---|
| 矩阵乘 | ✅ | `dotPacked4x8EXT` | 每 lane 持 16 B 权重 |
| softmax | ✅ | 标量循环 | 暂未向量化 |
| … | | | |

### 2. 设备硬边界

| 项 | 值 |
|---|---|
| 显存 / 共享内存总量 | |
| 单次分配上限 | |
| 单个 buffer 上限 | |
| 共享内存上限 | |
| 实测有效带宽 | |
| subgroup / warp 尺寸 | |

这些值会被写进编译期常量（`DeviceCaps`），**不做运行期探测**。

### 3. 最小验证程序

能算出 `rel_l2` 的那种：读一个真实 Q4_K 张量，跑一遍 GEMV，和 CPU 参考实现比对，打印 `rel_l2` 与有效带宽。

参考：`tests/sycl_mapping_check.cpp`（无需 GPU 的索引校验）与 `tests/sycl_bench.cpp`（上机基准）。

## 不接受的形态

- 缺 softmax 或矩阵乘，指望模型层用别的算子凑出来；
- 运行期探测能力、按能力分支的后端；
- 没有验证程序、只有“应该能跑”的声明。

## 合入之后

1. 在 `code/device/<backend>/` 落地；
2. 在 `code/kernel/<backend>/` 提供算子实现；
3. 在 README 的支持矩阵里加一行；
4. 发版时在 `FJEC_Vx.y.z` 的发行说明里注明新增后端。
