#!/bin/bash

# Geo-Sign Inference Script
# 使用三个模型检查点进行推理

# ============================================================================
# Mode 说明 (与训练时一致):
#   - hand: 仅手部数据 (left, right)
#   - hand_body: 手部+身体数据 (body, left, right)
#   - hand_body_face: 全部数据 (body, left, right, face_all)
# ============================================================================

# 模型检查点路径
CKPT1="/data/taoye/Geo-Sign/out/train_pure_hand_4D/checkpoint_49.pth"
CKPT2="/data/taoye/Geo-Sign/out/train_hand_body_4D/checkpoint_49.pth"   # 待训练
CKPT3="/data/taoye/Geo-Sign/out/train_hand_body_face_4D/checkpoint_49.pth"

# 模型名称
NAME1="hand"
NAME2="hand_body"
NAME3="hand_body_face"

# 每个模型对应的数据模式
MODE1="hand"
MODE2="hand_body"
MODE3="hand_body_face"

# 输出目录
OUTPUT_DIR="./inference_results"

# 数据集
DATASET="CSL_Daily"

# 运行推理
python inference.py \
    --checkpoint_paths "$CKPT1" "$CKPT2" "$CKPT3" \
    --checkpoint_names "$NAME1" "$NAME2" "$NAME3" \
    --modes "$MODE1" "$MODE2" "$MODE3" \
    --dataset "$DATASET" \
    --batch_size 8 \
    --num_beams 4 \
    --max_new_tokens 100 \
    --output_dir "$OUTPUT_DIR" \
    --save_predictions

# ============================================================================
# 测试模式 (少量样本)
# ============================================================================
# python inference.py \
#     --checkpoint_paths "$CKPT1" "$CKPT2" "$CKPT3" \
#     --checkpoint_names "$NAME1" "$NAME2" "$NAME3" \
#     --modes "$MODE1" "$MODE2" "$MODE3" \
#     --dataset "$DATASET" \
#     --num_samples 10 \
#     --output_dir "$OUTPUT_DIR" \
#     --save_predictions
