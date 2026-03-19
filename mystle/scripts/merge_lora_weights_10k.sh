python scripts/merge_lora_weights.py \
    --model-path /root/fsas/zhanghongyu/LLaVA/models/checkpoints_drivelm_qlora_10k/checkpoint-650 \
    --model-base /root/fsas/models/LLaVA/llava-v1.6-vicuna-7b \
    --save-model-path /root/fsas/zhanghongyu/LLaVA/models/drivelm_llava_merged/finetune_lora_10k \