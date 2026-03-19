python scripts/merge_lora_weights.py \
    --model-path /root/fsas/zhanghongyu/LLaVA/models/checkpoints_drivelm_qlora_1k/checkpoint-60 \
    --model-base /root/fsas/models/LLaVA/llava-v1.6-vicuna-7b \
    --save-model-path /root/fsas/zhanghongyu/LLaVA/models/drivelm_llava_merged/finetune_lora_1k \