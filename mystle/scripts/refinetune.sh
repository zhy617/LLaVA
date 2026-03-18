orig=/root/LLaVA/llava/train/train.py; \
bak="$(mktemp /tmp/train.py.bak.XXXXXX)"; \
cp "$orig" "$bak"; \
trap 'mv "$bak" "$orig"' EXIT; \
sed -i 's/trainer.train(resume_from_checkpoint=True)/trainer.train(resume_from_checkpoint=False)/' "$orig"; \
rm -rf /root/fsas/zhanghongyu/LLaVA/models/checkpoints_drivelm_qlora/checkpoint-*; \
cd /root/LLaVA && bash ./mystle/scripts/finetune_lora.sh