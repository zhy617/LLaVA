#!/bin/bash

set -e

PYTHON_CMD=(python)
if ! "${PYTHON_CMD[@]}" -c "import torch" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME:-llava}" python)
    else
        echo "No usable Python with torch found, and conda is unavailable." >&2
        exit 1
    fi
fi

"${PYTHON_CMD[@]}" /root/LLaVA/mystle/finetune_drivelm.py \
    --model-path "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b" \
    --data-path "/root/fsas/dataset/OpenDriveLab/DriveLM/v1_1_train_nus.json" \
    --image-root "/root/fsas/dataset/OpenDriveLab/DriveLM/nuscenes/samples" \
    --output-dir /root/fsas/zhanghongyu/LLaVA/models/Finetuned_lora \
    --num-scenes 1 \
    --scene-seed 42 \
    --qa-groups perception behavior \
    --epochs 1 \
    --max-samples 200 \
    --learning-rate 2e-4 \
    --grad-accum-steps 4 \
    --lora-enable \
    --load-4bit \
    --gradient-checkpointing
