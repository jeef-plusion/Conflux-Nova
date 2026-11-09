# 交接：nova_v2 SYCL 后端 · 目标机 Arc 130T（方案 C 待测）

> 写于 2026-11-09。给「另一个对话」直接读这一份，不需要再粘贴上下文。
> 读之前先记住一句话：**代码已落地、数值已验完，唯一没做的是「在 Arc 上量带宽」，而这台机器
> 没有散热器，量法错了数字就自相矛盾。**
>
> ⚠️ 第 2 节含明文口令。这份文件只适合放私有工作区；要进公开仓库先把第 2 节删掉。

---

## 1. 现在要做的唯一一件事

**没有待测的了。方案 C 已经在目标机上测完并被否决（2026-11-09，见 §6）。**
默认值已回退到 `NOVA_SYCL_GEMV_SPLIT=1`，仓库当前状态就是「可发布」的配置。

如果要复现那次结论，或者接着追剩下的 40% 带宽，用这一条命令（参数一个都不能省，每个都在
防一个已实测到的坑）：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cd /d C:\nova
build_sycl.bat 1.5b
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf --iters 300 --ab-iters 2000 --ab-cooldown 0 --warm-ms 2000 --no-cold
```

看两处，顺序不能换：

| 顺序 | 看什么 | 判据 |
|---|---|---|
| 1 | §3 / §3b 的 `rel_l2` | 全 < 1e-4（预期 1e-7 量级）。有一行爆了就停，别看带宽 |
| 2 | §4c 的「每臂每次测量时长」和四臂相对差 | 时长必须 ≥ 50 ms；差值 > 2σ 才算数；§4d 的漂移 > 10% 时只认相对差 |

测得的两条硬事实（写下来免得下次再踩）：

- **GEMV 稳态 = 36.8 GB/s**（不是 18.27）＝ 61.38 的 60%。「只有 30%」是冷态+并发+短测量的假象；
- **方案 C 三个臂全部显著更慢**：4 段连续 -2.7%、2 段 -14.1%、4 段交错 -7.5%。


---

## 2. 机器访问

主机 `192.168.137.8`（Win11 26200，主机名 `LAPTOP-O94L9GV5`），账号 `Administrator`，口令见密码管理器（**仓库内不写明文**）。

```powershell
# 从环境变量取，避免落进仓库或日志
$pw = ConvertTo-SecureString $env:ARC_PW -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential('Administrator', $pw)
```

### WinRM（执行命令）

```powershell
$pw   = ConvertTo-SecureString '<ARC_PASSWORD>' -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential('Administrator', $pw)
$so   = New-PSSessionOption -OperationTimeout 600000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -SessionOption $so -ScriptBlock {
    cmd /c "cd /d C:\nova && build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf --iters 100"
}
```

首次报信任错误时，在**发起端**执行：

```powershell
Start-Service WinRM
Set-Item WSMan:\localhost\Client\TrustedHosts -Value "192.168.137.8" -Force
```

注意：目标机在编译时 CPU 打满，WinRM 会话会**很慢甚至超时**（实测一次 2 分钟+）。
`OperationTimeout` 给到 600000 以上，或者用「写日志 + 轮询日志」的方式，不要让一次调用等太久。

### SMB（传文件，最可靠）

```powershell
net use \\192.168.137.8\C$ /user:Administrator "<ARC_PASSWORD>"
Copy-Item 本地文件 "\\192.168.137.8\C$\nova\..."
```

报 1219 先 `net use \\192.168.137.8\C$ /delete`。

### RDP

`mstsc /v:192.168.137.8`。

---

## 3. 路径与编译顺序

```
仓库      C:\nova
模型      C:\nova\build\q4k_15b.gguf        (Qwen2.5-1.5B Q4_K, 959.5 MB)
          C:\nova\build\q4k.gguf            (Qwen2.5-7B Q4_K)
oneAPI    C:\Program Files (x86)\Intel\oneAPI\setvars.bat    (icpx 2026.1.1)
MSVC      C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
CMake     C:\Tools\cmake\bin\cmake.exe
Git       C:\Tools\MinGit\cmd\git.exe
```

**先 `vcvars64.bat`，再 `setvars.bat`**，顺序反了链接失败。

### .bat 的换行问题（已修，别再用旧结论）

仓库里的 `.bat` 原来全是 **LF 换行**。在非 UTF-8 控制台代码页下 cmd 会把它们**拆行执行**
（症状：`build_sycl.bat` 被吃成 `d_sycl.bat`、`'ycl' 不是内部或外部命令`）。
2026-11-09 已把根目录所有 `.bat` 转成 **CRLF**，`build_sycl.bat` 现在能正常跑到
「找不到 icpx」那一步。**再出现同类症状，先查换行，不要先怀疑脚本逻辑。**
另外 `build_sycl_host.bat` 刻意写成纯 ASCII + 单行 + goto，就是为了绕开这个坑。

---

## 4. 硬件实测（不要抄文档里的旧数）

| 项 | 值 |
|---|---|
| device | Intel(R) Arc(TM) 130T GPU (16GB)，Core Ultra 5 225H |
| Xe cores / EU | 112，clock 2200 MHz |
| global mem | 16831 MiB |
| max alloc | **4095 MiB**（单次分配上限） |
| local mem | **64 KiB**（旧文档写 32 KiB，错） |
| max workgroup | 1024 |
| subgroup | 默认 8，支持 8 / 16 / 32 |
| USM | device / shared 都支持，fp16 支持 |
| 顺序读上限 | **61.38 GB/s**（`tools/bench_ab.exe`，512 MB 扫描，12 轮中位数） |

---

## 5. 测量协议（这台机器没有散热器 —— 彩蛋是「冷却」反而会害你）

**结论先行：在这台机器上，绝对值只由「测量有多长」和「机器当时什么状态」决定，方案之间的差别
（±3~14%）远小于状态带来的差别（最多 2.3 倍）。** 已实测的状态表：

| 状态 | us/次 | GB/s |
|---|---:|---:|
| 冷启动 + 5 次 warmup + 三个并发编译 | 82.50 | 16.09 |
| 冷启动，干净机器，第 1 批 20 次 | 49.3~51.0 | 26.0~26.9 |
| 提频到第 8 批 | 41.6 | 31.9 |
| 烧 2 s 后短测 300 次 | 38.5~50.1 | 26.5~34.5 |
| **2000 次长测量 × 3 轮** | **36.07** | **36.79** |
| 10 次发射 + 6 s 空闲（"冷却"） | 100 ~ 19745 | 掉 2~400 倍 |

所以正确的协议是（`sycl_bench` 已经把前四条做成默认值）：

| 做法 | 原因 |
|---|---|
| **只编译一次**，一个二进制里含全部四个臂 | 编译 2~3 分钟全核打满，是最大热源；重编一次等于换一个热状态 |
| `--warm-ms 2000` 先烧 2 s 再计时 | 冷启动比提频态慢 ~20% |
| `--ab-iters 2000`（≥70 ms/臂/轮） | 10 次发射 = 0.5 ms，落在提频/唤醒过渡段，测的是过渡态 |
| `--ab-cooldown 0` **不要空闲** | 空闲 6 s 会让 GPU 掉回 idle 时钟：实测 50 → 100 us，最坏一次 19.7 ms |
| 同一轮内四臂交替 | 四臂共享同一热状态，相对差才可比 |
| 同一时刻只跑一个进程，编译不与测量并行 | 三个并发 sweep 叠加时读数 16.09（干净环境 36.8） |
| 看 §4c「降频检查」与 §4d「热漂移复测」 | 这两个数直接回答「这次算不算稳态」；> 10% 时只认相对差 |

**「每轮之间 sleep 5-10 s」这条经验在本仓库的基准上是错的**，因为一次测量只有几十毫秒；
冷却只在「一整轮足够长（秒级）」时才有意义。

---

## 6. 方案 C：现状（已结案：输了）

改动（工作树里，未提交）：

| 文件 | 内容 |
|---|---|
| `src/kernel/sycl/q4k_gemv.cpp` | `GemvQ4K<K,HW,SPLITS,STRIPE>`：SPLITS 个 subgroup 分段扫同一行；段内仍是 8 lane 拼 128 B；4 个部分和走 `local_accessor` + 一次 `barrier` |
| `src/kernel/sycl/sycl_common.h` | `NOVA_SYCL_GEMV_SPLIT`（**默认 1 = 旧映射；实测 4 段更慢，已回退**）、`NOVA_SYCL_GEMV_STRIPE`（默认 0）、`q4k_seg_begin/end` |
| `src/kernel/sycl/kernels.h` | `gemv_variant<K,HW,SPLITS,STRIPE>`，实例化 1 / 2 / 4连续 / 4交错 |
| `tests/sycl_bench.cpp` | §4a 提频曲线、§3b 四臂对拍、§4c A/B（长测量 + 降频检查 + 时长守卫）、§4d 热漂移复测、`--warm-ms/--ab-iters/--ab-cooldown` |
| `tests/sycl_gemv_host_check.cpp` + `tests/host_sycl/sycl/sycl.hpp` | **不需要 GPU**：垫片把真 kernel 跑在 CPU 上对拍 |
| `build_sycl_host.bat` | 上面那个对拍的构建入口（只要 MSVC） |

**已验证（CPU 上跑真 kernel，不是复刻）**：`rel_l2` 全部 1e-7 量级，四个臂全 PASS；
1.5B/7B × SG16/32 四种组合都过；尾部守卫（rows 不是每 WG 行数整数倍）也过。
复现：`build_sycl_host.bat 1.5b 16` → `build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf`。

**目标机实测（2026-11-09，已结案）**：

| 臂 | 中位数 | 带宽 | vs 基线 | 判定 |
|---|---:|---:|---:|---|
| SPLITS=1（旧映射，**现为默认**） | 36.07 us | **36.79 GB/s** | — | 基线 / 最优 |
| SPLITS=2 连续段 | 41.99 us | 31.61 GB/s | -14.1% | 显著变慢 |
| SPLITS=4 连续段（方案 C） | 37.06 us | 35.81 GB/s | -2.7% | 显著变慢 |
| SPLITS=4 交错段 | 38.99 us | 34.04 GB/s | -7.5% | 显著变慢 |

（2000 次发射/臂/轮，3 轮，同进程交替，σ<1%，降频检查 -1.5%；详见 `PERF_BASELINE.md` §8）

**顺带纠正基线**：GEMV 稳态是 **36.8 GB/s**（61.38 的 60%），不是 18.27（30%）。
1.5B decode 推算相应从 20.36 → **41 tok/s**。「25 tok/s 目标」旧映射早就过了。


---

## 7. 为什么会输（事后对照，三条都命中了）

1. **PD 掉了**：`PD = min(kPrefetchDepth, NP)`。1.5B 上 NP 从 3 降到 1（每 lane 在飞 48 B → 16 B），
   7B 从 7 降到 2（64 B → 32 B）。`SPLITS=2` 是留的对照臂，结果它最差（-14.1%），说明
   「段内连续长度」和「在飞字节」两个变量都在往下走，叠起来就是负收益。
2. **1.5B 是最不友好的形状**：BPR=6 切成 4 段是 1/2/1/2，段长 1 的 subgroup 有一半 lane 组空转。
   7B（BPR=14）切成 3/4/3/4，浪费小得多 —— 所以这条没被 1.5B 的实测完全证否，7B 上没测。
3. **块头空洞**：每 128 B 之后有 16 B 的 `d/dmin/scales`，占 11% 的字节。方案 C 完全没动它。
4. 旧文档里「一行 1536 字节、24 个 super-block、相邻 lane 隔 144 B」是错的：
   1.5B 一行只有 6 块 / 864 B，7B 是 14 块 / 2016 B；subgroup 内部是 8 lane 拼一条 128 B。
   已在 `docs/SYCL_PORT.md` §11.1 更正。

---

## 8. 绝对不要做的事

- **不要**为每一档重新编译再比（编译 2~3 分钟全核打满，是最大热源）；
- **不要**在测量之间 sleep「冷却」——短测量 + 空闲 = 测到 idle 时钟（实测 50 → 100 us，最坏 19.7 ms）；
- **不要**用 10 次发射这种短测量下结论（0.5 ms，进不了稳态）；
- **不要**让两个基准/编译同时跑（并发污染实测 16.09 vs 36.8）；
- **不要**在没有 `rel_l2` 校验的情况下判断性能提升（SG=16 那次 attention 爆到 2.1e38）；
- **不要**用 `malloc_shared` 替 `malloc_device`（实测差 8.6%，标准差 9.8%，在噪声内）；
- **不要**做 Kernel Fusion（dispatch 5 µs/次 × 14 次/层 × 28 层 = 1.96 ms，只占 2.7%）；
- **不要**用 LF 换行保存 `.bat`（已修，别改回去）。

---

## 9. 相关文档

| 文件 | 内容 |
|---|---|
| `docs/PERF_BASELINE.md` | 基线数字 + §7 测量状态的坑（提频/并发/冷却）+ §8 方案 C 的 A/B 结论 |
| `docs/SYCL_PORT.md` §11 | 方案 C 的设计、代码形状、代价分析、§11.5 验收协议、§11.6 实测结论 |
| `docs/SYCL_PORT.md` §9 | 本机怎么验的（§9.1 复刻索引算术、§9.2 跑真 kernel） |
| `docs/BACKEND_INTEL_255H.md` | 驱动/后端历史坑（int8 dot 的 device lost、错编译） |
