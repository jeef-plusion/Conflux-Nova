// code/kernel/1080ti/gemv_q4k.cu — 计算 TU（GEMV/MMQ + RMSNorm/RoPE/elementwise）
// 数值实现单点维护在 src/kernel/，此处按项目规范聚合为一个计算 TU。
#include "../../../src/kernel/q4k_gemv.cu"
#include "../../../src/kernel/norm.cu"
#include "../../../src/kernel/rope.cu"
#include "../../../src/kernel/elem.cu"
#include "../../../src/kernel/mmq.cu"
