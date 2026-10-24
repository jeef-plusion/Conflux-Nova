# 支持范围 · V2.0.0-Core

## 受支持
- 本仓库 `main` 分支上的 V2.0.0-Core 源码与构建脚本。
- 五个编译期模具：Qwen2.5-7B / 1.5B、DeepSeek-R1-Distill-Qwen-7B / 1.5B / Llama-8B。
- 纯 Q4_K GGUF（stock Q4_K_M 需先用 `tools/requant_q4k.cpp` 转换）。

## 不受支持（一律不响应）
- V2.0.0-Core 之前的任何代码、分支、tag、PR、Issue。
- Neo 五层分层架构（graph / device / chat / user / kernel 包装）相关的任何改动。
- v1.x 的性能数据、构建方式、以及与旧接口的兼容性诉求。

## 为什么
V2.0.0-Core 是一次底层重写：单体架构、编译期模具、无运行期 dispatch。
它与旧版本没有共享的控制流，也不存在渐进迁移路径。保留兼容层只会让新架构退化，
所以旧线整体封存。
