from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import warnings
# 过滤掉关于 "meta parameter" 的特定警告
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# ================= 修改开始 =================

# 1. 这里的路径改成你刚才下载的绝对路径
model_path = "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b"

# 2. 提示词 (Prompt)
prompt = "What are the things I should be cautious about when I visit here?"

# 3. 图片路径 (可以用 URL，也可以是你本地的图片路径 "/root/test.jpg")
image_file = "https://llava-vl.github.io/static/images/view.jpg"

# ================= 修改结束 =================

# 构造模拟参数对象
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    # 这里建议手动指定 model_name，防止自动解析出错
    "model_name": "llava-v1.6-vicuna-7b", 
    "query": prompt,
    # 强制指定对话模式，vicuna-7b 对应的模式通常是 "vicuna_v1"
    "conv_mode": None, 
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

# 运行推理
eval_model(args)