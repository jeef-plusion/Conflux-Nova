# B 站演示脚本 · Conflux Nova FJEC_V2.1.0

> 目标时长 4~5 分钟。五个分镜，全部实拍终端，不剪辑数字。

## 录制前准备

| 项 | 建议 |
|---|---|
| 终端 | Windows Terminal，字号 16~18，深色主题 |
| 分辨率 | 1920×1080，60fps |
| 模型 | `build/q4k.gguf`（7B）与 `build/abl15b_q4k.gguf`（1.5B） |
| 预热 | 先空跑一次 `nova_core.exe`，让权重进显存、时钟进 P2 |
| 频率 | 另开窗口跑 `nvidia-smi --query-gpu=clocks.sm,pstate -l 1`，出镜更有说服力 |

---

## 分镜 1 · 启动（15 秒）

```
build\nova_core.exe
```

> 这是 Conflux Nova，一个只跑在 1080 Ti 上的单体推理引擎。一个二进制就是一个模型，没有 graph 层，没有运行期 dispatch。

画面重点：ASCII 艺术字、`[INFO] Conflux Nova V2.0.0`、模型清单、池占用三行。

---

## 分镜 2 · 单路 55 t/s（60 秒）

```cpp
inline constexpr const char* kModelPath = "build/q4k.gguf";
inline constexpr const char* kCardName  = "Qwen2.5-7B-Instruct";
```

```
build\nova_core.exe build\q4k.gguf "请推导 f(x)=x^3-3x 的极值点"
```

> 7B，Q4_K，2048 上下文。prefill 536 tok/s，decode 55 tok/s。
> 注意它是逐字吐出来的——输出链是流式的，不是等全部生成完再打印。

定格：`[PERF] decode … 55.24 tok/s`

---

## 分镜 3 · 豆腐脑吵架（90 秒）

三个角色，`kSys[3]`：

```cpp
"你是甜党死忠，坚信豆腐脑必须放糖，语气激动，爱引古诗。50字以内。"
"你是咸党死忠，只认酱油榨菜，嘲笑放糖的是异端。50字以内。"
"你是乐子人，主张放辣椒，疯狂拱火。50字以内。"
```

> 三个完全不同的 AI，同一张卡，各自持有自己的 KV 和人格。
> 每一轮，它们都会读到对方上一轮说了什么。

台词预期：

```
[甜党]   豆腐脑必放糖，如诗云：甜如蜜，幸福蜜。
[咸党]   放糖的是异端，必须是酱油榨菜组合！
[乐子人] 放辣椒！辣才是王道！
```

---

## 分镜 4 · 四大天王辩论（120 秒）

四个角色，`kSys[4]`：咸粽党 / 甜粽党 / 辣椒邪教 / 不爱吃。

> 四路并发。gemv4 让四个会话共享一次权重读取——这是它能跑起来的关键。
> 注意最后那行 PERF：四路加起来 104 tok/s。

```
[A 咸粽党]  咸粽才是正宗！蛋黄五花肉才是灵魂！
[B 甜粽党]  甜粽才是王道，豆沙蜜枣才是经典！
[C 辣椒邪教] 我都反对，应该咸甜混合加辣椒芝士榴莲！
[D 不爱吃]  都不想吃，粽子对我来说就是碳水。
[PERF] round 1: 32 tok x4 in 1219 ms | 26.2 tok/s per session | 105.0 tok/s total
```

彩蛋：让观众看第三轮——不爱吃会从一整句缩成四个字「非吾所好」。

---

## 分镜 5 · 收尾（20 秒）

> 架构叫 Conflux Nova Neo 2.0.0-Core，这一版发行叫 FJEC_V2.1.0。
> MIT 协议，欢迎提 PR——我们接受的是模块，不是新架构。

```
github.com/jeef-plusion/Conflux-Nova
FJEC_V2.1.0 · MIT License
```

---

## 录制检查清单

- [ ] 跑之前先空跑一次预热，避免第一轮被 P5 拖慢
- [ ] 同时录一份 nvidia-smi 频率窗口，数字更可信
- [ ] 分镜 3、4 各录两遍，挑人格最稳的那次
- [ ] 别剪掉 `[PERF]` 行——那是这项目的说服力所在
