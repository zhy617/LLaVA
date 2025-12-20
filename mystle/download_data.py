import os
from huggingface_hub import snapshot_download

# 1. 必不可少的镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 配置参数
repo_id = "OpenDriveLab/DriveLM"
# 建议放在数据盘，比如 /root/autodl-tmp/datasets/DriveLM
local_dir = "/root/fsas/datasets/OpenDriveLab/DriveLM" 

print(f"准备下载数据集: {repo_id}")
print(f"保存路径: {local_dir}")

# 3. 开始下载
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    repo_type="dataset",           # <--- 关键修改：必须指定类型为 dataset
    local_dir_use_symlinks=False,  # 下载真实文件
    resume_download=True,
    max_workers=2,
)

print("数据集下载完成！")