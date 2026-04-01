#!/bin/bash

set -u

# =========================
# 配置区
# =========================

GPU_ID=1
CHECK_INTERVAL=120
UTIL_THRESHOLD=25
MEM_THRESHOLD=1000
REQUIRE_CONSECUTIVE=5

TRAIN_SCRIPT="/data/taoye/Geo-Sign/script/train_hand_body.sh"
LOG_FILE="/data/taoye/Geo-Sign/out/wait_train.log"

# 固定训练使用这张卡
export CUDA_VISIBLE_DEVICES=$GPU_ID

# =========================
# 基础检查
# =========================

mkdir -p /data/taoye/Geo-Sign/out

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] Error: nvidia-smi not found." | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[$(date '+%F %T')] Error: train script not found: $TRAIN_SCRIPT" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -x "$TRAIN_SCRIPT" ]; then
    chmod +x "$TRAIN_SCRIPT"
fi

echo "[$(date '+%F %T')] Start monitoring GPU $GPU_ID ..." | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] Condition: util < ${UTIL_THRESHOLD}%, mem_used < ${MEM_THRESHOLD}MB for ${REQUIRE_CONSECUTIVE} consecutive checks" | tee -a "$LOG_FILE"

idle_count=0

while true; do
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F ', ' -v gpu="$GPU_ID" '$1==gpu {print $2" "$3}')

    if [ -z "$gpu_info" ]; then
        echo "[$(date '+%F %T')] Warning: cannot read GPU $GPU_ID info" | tee -a "$LOG_FILE"
        sleep "$CHECK_INTERVAL"
        continue
    fi

    gpu_util=$(echo "$gpu_info" | awk '{print $1}')
    mem_used=$(echo "$gpu_info" | awk '{print $2}')

    echo "[$(date '+%F %T')] GPU $GPU_ID status: util=${gpu_util}%, mem=${mem_used}MB" | tee -a "$LOG_FILE"

    if [ "$gpu_util" -lt "$UTIL_THRESHOLD" ] && [ "$mem_used" -lt "$MEM_THRESHOLD" ]; then
        idle_count=$((idle_count + 1))
        echo "[$(date '+%F %T')] Idle check passed (${idle_count}/${REQUIRE_CONSECUTIVE})" | tee -a "$LOG_FILE"
    else
        idle_count=0
        echo "[$(date '+%F %T')] GPU still busy, continue waiting..." | tee -a "$LOG_FILE"
    fi

    if [ "$idle_count" -ge "$REQUIRE_CONSECUTIVE" ]; then
        echo "[$(date '+%F %T')] GPU $GPU_ID seems idle. Launch training now." | tee -a "$LOG_FILE"
        bash "$TRAIN_SCRIPT" >> "$LOG_FILE" 2>&1
        echo "[$(date '+%F %T')] Training script exited." | tee -a "$LOG_FILE"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done