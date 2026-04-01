#!/bin/bash

# Geo-Sign Training Script
# 训练 hand+body 模式 (手部 + 身体，不含面部)

# ============================================================================
# Mode 说明:
#   - hand: 仅手部数据 (left, right)
#   - hand_body: 手部+身体数据 (body, left, right)  ← 当前脚本
#   - hand_body_face: 全部数据 (body, left, right, face_all)
# ============================================================================

# 输出目录
output_dir=out/train_hand_body_4D

# 预训练权重 (如果有的话)
ckpt_path=checkpoints/pretraining.pth

# 训练参数
deepspeed --include localhost:1 --master_port 29511 fine_tuning.py \
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
  --mode hand_body \
  --input_channels 4 \
  --max_length 256 \
  --seed 42

# ============================================================================
# 注意事项:
# 1. --mode hand_body 表示只使用手部+身体数据
# 2. 如果预训练权重不匹配，会自动跳过并从随机权重开始训练
# 3. 如果要评估，添加 --eval 参数
# ============================================================================

# 评估命令示例:
# deepspeed fine_tuning.py \
#   --deepspeed \
#   --batch_size 8 \
#   --output_dir $output_dir \
#   --finetune $output_dir/best_checkpoint.pth \
#   --dataset CSL_Daily \
#   --task SLT \
#   --mode hand_body \
#   --input_channels 4 \
#   --eval
