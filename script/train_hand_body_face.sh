#!/bin/bash

# Geo-Sign Training Script
# 训练 hand_body_face 模式 (全部数据: 手部+身体+面部)

# ============================================================================
# Mode 说明:
#   - hand: 仅手部数据 (left, right)
#   - hand_body: 手部+身体数据 (body, left, right)
#   - hand_body_face: 全部数据 (body, left, right, face_all)  ← 当前脚本
# ============================================================================

# 输出目录
output_dir=out/train_hand_body_face_4D

# 预训练权重 (如果有的话)
ckpt_path=checkpoints/pretraining.pth

# 训练参数
deepspeed fine_tuning.py \
  --deepspeed \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 50 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --mode hand_body_face \
  --input_channels 4 \
  --max_length 256 \
  --seed 42
