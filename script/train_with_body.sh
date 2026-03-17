deepspeed --include localhost:1 fine_tuning.py \
  --deepspeed \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 50 \
  --output_dir out/body_4d_train \
  --dataset CSL_Daily \
  --input_channels 4 \
  --finetune checkpoints/pretraining.pth \
  --task SLT \
  2>&1 | tee out/body_4d_train/training.log
