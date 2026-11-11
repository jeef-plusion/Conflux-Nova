# 贡献者指南 · CONTRIBUTING

> 这个仓库的代码要能被陌生人**在半小时内读懂、在一小时内改对**。
> 下面每条都是硬性要求：不满足的 PR 不会被合，哪怕它跑得再快。

---

## 1. 一个 PR 必须包含三样东西

缺任何一样，Review 直接打回。

### 1.1 功能特点（What）

一句话说清"这个 PR 让什么东西变得更好/更快/更正确"，**带数字**。

```
❌ 优化了 GEMV 性能
✅ GEMV 在 Arc 130T 上从 34.2 → 37.4 GB/s（+9.4%），端到端 34.23 → 36.32 tok/s

❌ 修了 bug
✅ 修掉 SG=16 时 quant 只量化半个 32-元素块的问题（int8 不一致 3327 → 0）
```

### 1.2 核心参数详解（Tunables）

只要你的改动引入了**任何**编译期常量、环境变量、命令行参数，必须列表说明：
名称、类型、默认值、**取值范围**、默认值为什么是这个值、改了会怎样。

```markdown
| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `NOVA_SYCL_GEMV_PD` | 4 | 1..8 | 预取窗深度。实测 4 最优；8/16 因寄存器溢出到私有内存反而变慢（-18%/-40%） |
| `NOVA_SYCL_SG` | 16 | 8/16/32 | subgroup 大小。必须整除 32（Q4_K 子块是 32 元素），改前先看 static_assert |
```

### 1.3 逻辑流程说明（How）

用编号步骤写清数据怎么流动，**至少覆盖一次完整的输入→输出**。涉及 kernel 的必须写清
lane/work-group 的映射关系。

```
1. host: 读 GGUF 张量 → 主机缓冲（AoS，144 B/块）
2. host: repack 成 SoA（qs 128 B/块连续 + hdr 16 B/块，按内核访问顺序排）
3. device: 每个 subgroup 负责一行；8 lane 各持 16 B qs；4 个块头一次 64 B 载入
4. device: 每 lane 累加 6 个 super-block → sg_sum → lane0 写 out[row]
```

---

## 2. 禁止出现（Review 会当场拦下）

| 禁止 | 为什么 | 正确写法 |
|---|---|---|
| **魔数** | `144`、`128`、`16`、`256` 散落在代码里，改一个格式就全错 | 走 `kBlockBytes` / `kQK` / `kQsBytes` 等常量，并在常量处注明推导 |
| **无意义命名** | `a, b, tmp, data2, flag` | `act_q8`、`wrow`、`row_bytes`、`is_first_tensor` |
| **隐式内存泄漏** | `cudaMalloc`/`sycl::malloc_device` 没有对应释放；异常路径漏掉 `destroy()` | RAII 或成对的 `create/destroy`；在析构里写清楚"谁负责释放" |
| **直接吞错误** | `cudaGetLastError()` 不查、`catch (...)` 空实现 | 一律 `NOVA_CU(...)`；SYCL 用 `wait_and_throw()`；确实要忽略时写注释说明为什么安全 |
| **无注释的边界技巧** | `(sbi < hi ? sbi : hi - 1)` 这种"越界读最后一块" | 必须写："越界那几次读的是本段最后一块，结果会被丢弃，避免读到行外" |
| **未说明的裸指针** | `float* qb_` 指向谁、活多久、谁释放 | 成员注释里写"指向 arena 的偏移，arena 析构后失效；生命周期 = E2ERunner" |

## 3. 必须包含

1. **边界检查的注释**：每一处 `if (row >= rows_)`、`hi - 1`、`min/max` 都要一句话说明"防的是什么"。
2. **指针生命周期的说明**：新增任何指针成员/参数，注释里写清**谁分配、谁释放、活到什么时候**。
3. **硬件差异说明**：同一份逻辑在 CUDA 与 SYCL 上行为不同时（lane 映射、subgroup 大小、
   是否有 `dotPacked4x8`、TDR 风险），在注释里写明差异点，别让下一个人重新踩。

---

## 4. 提交前自检（跑完这四条才算完）

```bat
:: 1) CPU 垫片数值对拍（不需要 GPU）
build_sycl_host.bat 1.5b 16
build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf     :: 必须 失败项 = 0

:: 2) 探针自检
build_scope_host.bat && build\scope_host_check.exe               :: 必须 失败项 = 0

:: 3) 性能（在 Arc 机器上，同一协议）
pd_sweep.bat 4                                                   :: 记录 ms/token 与各 op 带宽

:: 4) 端到端
build_sycl_e2e.bat 1.5b && build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 16
```

**性能类 PR 必须附 A/B 前后对比**，且必须用 `pd_sweep.bat` 的同一套协议
（长测量、单进程、`--warm-ms 2000`）。只贴一个"快了 20%"的截图不算。

---

## 5. 代码风格（不讨论，照做）

- 缩进 4 空格，不用 tab；行宽 ≤ 110；
- 命名：类型 `PascalCase`，函数 `snake_case`，常量 `kPascalCase`，成员 `name_`；
- 每个 `.cpp`/`.h` 顶部必须有 3~8 行的**文件级注释**：它解决什么问题、与哪个文件对应、有什么坑；
- 注释用中文，标识符用英文；**注释解释"为什么"，不解释"是什么"**（代码已经说了是什么）；
- 不允许 `using namespace` 出现在头文件里。

---

## 6. Review 流程

1. 作者自查（第 4 节四条命令全绿 + 自评清单）；
2. **至少一名**审查者按三个维度看：内核（对齐/越界/同步）、架构（生命周期/所有权）、工程（构建/命名）；
3. 有 P0 问题（会崩/会算错）直接打回，不看性能；
4. 合并前 squanch 成一个 commit，message 用 `type(scope): 一句话`（见下）。

```
feat(sycl): ...     新功能
fix(quant): ...     修 bug
perf(gemv): ...     性能
docs(sycl): ...     文档
test(scope): ...    测试
chore: ...          杂项
```

---

## 7. 不要做的事

- **不要**提交二进制（`release/*.exe`、`*.dll` 已在 `.gitignore`；走 GitHub Releases）；
- **不要**修改 `build_sycl.bat` 等构建脚本里的路径而不更新 `docs/LAYOUT.md`；
- **不要**在 `core` 的编译清单里加入任何探针 `.cpp`（探针只能是头文件、编译期可关）；
- **不要**为了一个 op 的 1% 收益引入新的全局状态；
- **不要**在没有 `rel_l2` 对拍的情况下声称"数值一致"。
