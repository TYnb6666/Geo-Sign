deepspeed --include localhost:1 fine_tuning.py \
  --deepspeed \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 50 \
  --output_dir out/train_hand_body_face_4D \
  --dataset CSL_Daily \
  --input_channels 4 \
  --finetune checkpoints/pretraining.pth \
  --task SLT \
  2>&1 | tee out/train_hand_body_face_4D/training.log

# 可选：从头训练（不使用预训练权重）
# deepspeed --include localhost:1 fine_tuning.py \
#   --deepspeed \
#   --batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --epochs 50 \
#   --output_dir out/train_hand_body_face_4D_scratch \
#   --dataset CSL_Daily \
#   --input_channels 4 \
#   --task SLT \
#   --lr 1e-4 \
#   2>&1 | tee out/train_hand_body_face_4D_scratch/training.log

# 可选：使用更小的 batch size 进行调试
# deepspeed --num_gpus 1 fine_tuning.py \
#   --deepspeed \
#   --output_dir out/debug_hand_body_face_4D \
#   --finetune checkpoints/pretraining.pth \
#   --dataset CSL_Daily \
#   --task SLT \
#   --batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --input_channels 4 \
#   --epochs 10 \
#   --lr 1e-4 \
#   2>&1 | tee out/debug_hand_body_face_4D/train_log.txt
