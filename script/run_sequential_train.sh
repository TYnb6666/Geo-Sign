#!/bin/bash

# 顺序运行训练脚本
# 1. 先运行 train_with_body.sh (Hand + Body)
# 2. 后运行 4d_hand_body_face.sh (Hand + Body + Face)

set -e  # 遇到错误时退出

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "开始顺序训练"
echo "=========================================="
echo ""

# 确保 out 目录存在
mkdir -p out/body_4d_train
mkdir -p out/train_hand_body_face_4D

# ==========================================
# 第一阶段: Hand + Body 训练
# ==========================================
echo "[1/2] 开始运行 train_with_body.sh (Hand + Body)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

bash "$SCRIPT_DIR/train_with_body.sh"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ train_with_body.sh 运行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✅ train_with_body.sh 完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 第二阶段: Hand + Body + Face 训练
# ==========================================
echo "[2/2] 开始运行 4d_hand_body_face.sh (Hand + Body + Face)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

bash "$SCRIPT_DIR/4d_hand_body_face.sh"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ 4d_hand_body_face.sh 运行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✅ 4d_hand_body_face.sh 完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "=========================================="
echo "🎉 所有训练任务完成！"
echo "=========================================="
echo "输出目录:"
echo "  - Hand+Body:      out/body_4d_train/"
echo "  - Hand+Body+Face: out/train_hand_body_face_4D/"
