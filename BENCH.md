> 仅对 V2.0.0-Core 生效。v2.0.0 之前的版本与 PR 不再支持。

# V2.0.0-Core · 1080 Ti 实测记录

## 结果（ctx=2048, 生成 128）

| | Qwen2.5-7B Q4_K_M |
|---|---|
| prefill 2048 tok | **470.4 tok/s**（4.35 s） |
| decode | **39.52 tok/s**（25.3 ms/tok） |
| gemv/matmul | 73.6% / 76.9% |
| attention | 22.3% / 8.3%（1.95 ms/tok） |
| 显存峰值 | 7083 MB |

演进：prefill 212.7 → 470.4（2.2×），decode 15.49 → 39.52（2.55×）。

## 两刀

**Attention**（`attn.cu`）：一个 block 的 4 个 warp 各扫 1/4 KV 段，各自做
online-softmax 局部和，再在 block 内用 smem 合并成一份 partial。
decode 的 attention 7.2 → **1.95 ms/tok**（3.7×），占比 24.6% → 8.3%。

**GEMV**（`q4k_gemv.cu`）：warp 一行，lane 8 个权重，三级寄存器流水线（q0/q1 + sb+2 预取）。

## 没做到的

- GEMV 目标 290 GB/s，实测 **224 GB/s**（4.05 GB / 18.1 ms）。
- 128-bit 权重 + 预算和 的版本写完跑出 illegal memory access，
  定位没跑完（isolated 各形状都通过，进引擎在 q_proj 报错），已回退到已验证版本。
- 时钟陷阱：同一份代码在 P5 档会测出 98 ms/tok，顶到 P2 才是 25 ms/tok —— 数据必须同会话背靠背比。
