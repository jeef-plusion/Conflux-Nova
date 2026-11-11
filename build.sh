#!/usr/bin/env bash
# ============================================================================
#  build.sh - 统一构建入口（Linux / macOS，内部只调 CMake）
#
#  用法:
#     ./build.sh                      # 默认 CUDA + 7B
#     ./build.sh host                 # 只编 CPU 垫片（CI / 无显卡机器）
#     ./build.sh cuda 1.5b
#     ./build.sh cuda 7b clean        # 先删 build-cmake
#
#  后端: cuda | sycl | host      模型: 7b | 1.5b | dsv2
#
#  说明：SYCL 后端目前只在 Windows + oneAPI 上验证过；Linux 需要装 oneAPI 的
#       DPC++（icpx）并 source setvars.sh，否则请用 host。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-cuda}"
MODELARG="${2:-7b}"
CLEAN="${3:-}"

case "$BACKEND" in
    cuda) NOVA_BACKEND=CUDA ;;
    sycl) NOVA_BACKEND=SYCL ;;
    host) NOVA_BACKEND=HOST ;;
    *) echo "未知后端: $BACKEND（可选 cuda|sycl|host）" >&2; exit 2 ;;
esac

case "$MODELARG" in
    7b)   NOVA_MODEL=QWEN25_7B ;;
    1.5b) NOVA_MODEL=QWEN25_1P5B ;;
    dsv2) NOVA_MODEL=DEEPSEEK_V2_LITE ;;
    *) echo "未知模型: $MODELARG（可选 7b|1.5b|dsv2）" >&2; exit 2 ;;
esac

command -v cmake >/dev/null 2>&1 || { echo "找不到 cmake（需要 3.20+）" >&2; exit 2; }



echo "[build] backend=$NOVA_BACKEND model=$NOVA_MODEL"
cmake -B "$BDIR" -DNOVA_BACKEND="$NOVA_BACKEND" -DNOVA_MODEL="$NOVA_MODEL"
cmake --build "$BDIR" -j "$(nproc 2>/dev/null || echo 4)"

echo
echo "[build] 数值闸门（不需要显卡）:"
echo "    ctest --test-dir $BDIR --output-on-failure"
