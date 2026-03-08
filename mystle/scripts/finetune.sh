python /root/LLaVA/mystle/finetune_drivelm.py \
    --model-path "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b" \
    --data-path "/root/fsas/dataset/OpenDriveLab/DriveLM/v1_1_train_nus.json" \
    --image-root "/root/fsas/dataset/OpenDriveLab/DriveLM/nuscenes/samples" \
    --output-dir /root/fsas/zhanghongyu/LLaVA/models/Finetuned  \
    --num-scenes 1  \
    --scene-seed 42  \
    --qa-groups perception behavior  \
    --epochs 1  \
    --max-samples 200  \
