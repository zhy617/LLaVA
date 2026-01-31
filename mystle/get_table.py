import torch
import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# from transformers.generation.utils import compute_transition_scores
import os
import json
# import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 关键：在所有其他导入之前，设置默认的 attention 实现
from transformers.utils import is_flash_attn_2_available
# 注意：这里我们直接修改 LlamaConfig 的 _attn_implementation 默认值
# 这比 from_pretrained("dummy", ...) 的方法更直接
if is_flash_attn_2_available():
    print("Flash Attention 2 is available. Setting it as default for Llama.")
    from transformers import LlamaConfig
    LlamaConfig._attn_implementation = "flash_attention_2"
else:
    print("Flash Attention 2 is not available.")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from typing import Dict, List

import warnings
import torch
from PIL import Image

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


def calibrate_threshold_table(
    args, 
    tokenizer, 
    model, 
    image_processor, 
    all_data, 
    data_root_path, 
    save_path="threshold_table.json",
    sample_limit=100  # 限制校准的场景数量，太大会跑很久
):
    print("===== Starting Calibration for Threshold Table =====")
    
    # 存储每个 token_id 出现过的所有概率值
    # key: token_id (int), value: list of probabilities (float)
    token_prob_collector = defaultdict(list)
    
    scene_count = 0
    
    # 遍历数据
    for scene_id, scene_data in tqdm(all_data.items(), desc="Calibrating"):
        if scene_count >= sample_limit:
            break
        scene_count += 1
        
        key_frames = scene_data.get("key_frames")
        if not key_frames: continue
            
        # 对每个场景的每一帧进行推理
        # 注意：校准时我们通常不使用记忆，而是看模型在单帧输入下的表现，
        # 或者使用标准的逐帧生成，这样收集到的概率最能代表“模型认为正确的概率”。
        for frame_id, frame_info in key_frames.items():
            image_paths_dict = frame_info.get("image_paths")
            if not image_paths_dict: continue

            # --- 1. 图片加载与 Prompt 构建 (复用你的逻辑) ---
            camera_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
            image_files = []
            valid_cameras = []
            for cam in camera_order:
                if cam in image_paths_dict:
                    rel_path = image_paths_dict[cam]
                    full_path = os.path.join(data_root_path, cam, os.path.basename(rel_path))
                    if os.path.exists(full_path):
                        image_files.append(full_path)
                        valid_cameras.append(cam)
            
            if not image_files: continue

            try:
                images = [Image.open(f).convert('RGB').resize((800, 450)) for f in image_files]
                image_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            except Exception as e:
                continue

            image_prompt_parts = [f"- {name}: {DEFAULT_IMAGE_TOKEN}" for name in valid_cameras]
            image_section = "The following images are provided from different camera angles:\n" + "\n".join(image_prompt_parts)
            prompt = f"{image_section}\n\n{args.initial_prompt}"
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_for_model = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            # --- 2. 运行生成并获取 Scores ---
            with torch.inference_mode():
                # 关键：设置 output_scores=True 和 return_dict_in_generate=True
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[img.size for img in images],
                    do_sample=True, # 校准时建议开启采样，覆盖更多可能的token
                    temperature=args.temperature if args.temperature > 0 else 1.0, # 如果原始是0，校准时建议稍微给一点点或者保持 greedy
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_scores=True,           # <--- 必须开启
                    # return_dict_in_generate=True  # <--- 必须开启
                )

            # --- 3. 计算概率 ---
            # 手动实现 compute_transition_scores 的功能
            # 1. 将 scores 元组堆叠成一个张量 (max_new_tokens, batch_size, vocab_size)
            stacked_scores = torch.stack(outputs.scores, dim=0)
            # 2. 调整维度为 (batch_size, max_new_tokens, vocab_size)
            stacked_scores = stacked_scores.permute(1, 0, 2)
            # 3. 计算 log probabilities
            log_probs = torch.nn.functional.log_softmax(stacked_scores, dim=-1)
            
            # 4. 提取生成的 token IDs (不包括输入的 prompt)
            generated_token_ids = outputs.sequences[:, input_ids.shape[-1]:]
            
            # 5. 使用 gather 从 log_probs 中精确地找到每个生成 token 对应的 log probability
            #    需要为 generated_token_ids 增加一个维度以匹配 log_probs 的维度
            transition_scores = log_probs.gather(dim=-1, index=generated_token_ids.unsqueeze(-1)).squeeze(-1)
            
            # 转换为概率 (exp)
            probs = torch.exp(transition_scores[0]) # 取 batch 0
            generated_tokens = outputs.sequences[0][input_ids.shape[1]:] # 去掉 input 部分
            
            # --- 4. 收集数据 ---
            for token_id, prob in zip(generated_tokens.cpu().tolist(), probs.cpu().tolist()):
                # 跳过特殊 token (如 pad, eos 等，视情况而定，一般建议保留 EOS)
                token_prob_collector[token_id].append(prob)

    # --- 5. 计算阈值表 ---
    print("Calculating statistics...")
    threshold_table = {}
    
    # 设定一个默认阈值，给没见过的 token 使用 (例如 0.01 或 全局平均值)
    default_threshold = 0.001 
    
    for token_id, prob_list in token_prob_collector.items():
        if len(prob_list) == 0: continue
        
        # 策略 A: 绝对最小值 (最严格，容易受异常值影响变太低)
        # min_val = min(prob_list)
        
        # 策略 B: 分位数 (推荐，例如取第 1% 或 5% 的位数值)
        # 这样可以忽略极少数的极端低概率情况
        min_val = float(np.percentile(prob_list, 1)) # 1st percentile
        
        # 保留4位小数节省空间
        threshold_table[int(token_id)] = round(min_val, 6)

    print(f"Collected stats for {len(threshold_table)} unique tokens.")
    
    # --- 6. 补全未见过的 Token (可选) ---
    # 如果希望 table 包含词表所有 token，可以用默认值填充
    for idx in range(tokenizer.vocab_size):
        if idx not in threshold_table:
            threshold_table[idx] = default_threshold

    # --- 7. 保存 ---
    with open(save_path, 'w') as f:
        json.dump(threshold_table, f)
    
    print(f"Threshold table saved to {save_path}")
    return threshold_table


def main():
    # --- 用户需要配置的路径 ---
    # 你的JSON标注文件路径
    json_file_path = "/root/fsas/dataset/OpenDriveLab/DriveLM/v1_1_val_nus_q_only.json" 
    # 你的数据根目录，用于拼接JSON中的相对图片路径
    # 例如，如果图片路径是 "../nuscenes/samples/..."，而JSON文件在 ".../DriveLM/annotations/"
    # 那么 data_root_path 应该是 ".../DriveLM/"
    data_root_path = "/root/fsas/dataset/OpenDriveLab/DriveLM/val_data" # 请修改为你的 nuscenes 数据集所在的根目录
    model_path = "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b"
    model_name = get_model_name_from_path(model_path)

    # 1. 加载模型 (一次性)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        # load_4bit=True # 如果显存足够，可以设为False以获得更快速度
    )

    if hasattr(model, 'is_first_frame'):
        model.is_first_frame = True
    else:
        # 如果模型没有这个属性，你可能需要添加它，或者师兄的代码有其他初始化方式
        print("Warning: model does not have 'is_first_frame' attribute. The continuous generation logic might not work.")

    model.config.image_aspect_ratio = 'square'

    # 2. 定义参数
    args = type('Args', (), {
        "conv_mode": "llava_v1",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "initial_prompt": "Suppose you are driving, and I'm providing you with six images captured by the car's front, front-left, front-right, back, back-left and back-right camera. First, generate a description of the driving scene which includes the key factors for driving planning, including the presence of obstacles and the positions and movements of vehicles and pedestrians and traffic lights. After description, please predict the behavior of ego vehicle, including exactly the driving direction(straight, turn left or turn right) and driving speed(slow, fast or normal)."
    })()

    # 3. 加载JSON文件并对每个序列进行推理
    try:
        with open(json_file_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    calibrate_threshold_table(args, tokenizer, model, image_processor, all_data, data_root_path)

if __name__ == "__main__":
    main()