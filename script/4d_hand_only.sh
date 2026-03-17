deepspeed --include localhost:1 fine_tuning.py \
  --deepspeed \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 50 \
  --output_dir out/train_v3_4D \
  --dataset CSL_Daily \
  --input_channels 4 \
  --finetune checkpoints/pretraining.pth \
  --task SLT \
  2>&1 | tee out/train_pure_hand_4D/training.log

# deepspeed --num_gpus 1 fine_tuning.py \
#     --deepspeed \
#     --output_dir out/ce_csl_4d_finetune \
#     --finetune out/hands_only_baseline/best_checkpoint.pth \
#     --dataset CSL_Daily \
#     --task SLT \
#     --batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --input_channels 4 \
#     --epochs 50 \
#     --lr 1e-4 \
#     2>&1 | tee out/ce_csl_4d_finetune/train_log.txt