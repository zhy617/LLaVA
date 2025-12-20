import os
from huggingface_hub import snapshot_download

# 1. 设置镜像 (这是替代 mirror="aliyun" 的标准做法)
# 在标准 transformers 环境中，必须通过环境变量设置镜像，否则下载极慢
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 配置参数
repo_id = "liuhaotian/llava-v1.6-vicuna-7b"
# 建议存放在空间大的数据盘，比如 autodl-tmp
local_dir = "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b" 

print(f"准备下载模型: {repo_id}")
print(f"保存路径: {local_dir}")

# 3. 开始下载
# snapshot_download 的优势：
# - 不会把模型加载到内存 (省去 15GB 内存占用，防止 OOM)
# - 支持多线程下载
# - 下载的是完整文件，不是缓存
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 重要：设为 False 才能得到真实的 .bin/.safetensors 文件
    resume_download=True,          # 支持断点续传，网断了重跑就行
    max_workers=8                  # 开启多线程加速
)

print("下载完成！Download complete.")