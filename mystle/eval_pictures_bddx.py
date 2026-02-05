import os
import json
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 关键：在所有其他导入之前，设置默认的 attention 实现
from transformers.utils import is_flash_attn_2_available
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

def run_inference_for_sequence(
    args, 
    tokenizer, 
    model, 
    image_processor, 
    context_len, 
    scene_id: str, 
    frame_list: List[str],  # 修改：直接接收文件名列表
    data_root_path: str,
    use_memory_idea: bool = False
):
    """
    对一个数据序列（一个视频的所有帧）进行连续推理
    """
    print(f"===== Processing Scene: {scene_id} (Memory Idea: {use_memory_idea}) =====")

    total_scene_time_start = time.time()
    total_generated_tokens = 0
    total_inference_time = 0

    # [修改] 关键帧排序：按文件名中的数字排序 (0.jpg, 1.jpg, ... 10.jpg)
    # 假设文件名格式为 "数字.jpg"
    try:
        sorted_frames = sorted(frame_list, key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # 如果文件名不是纯数字，回退到字符串排序
        sorted_frames = sorted(frame_list)

    last_output = None

    for i, frame_filename in enumerate(sorted_frames):
        print(f"--- Processing Frame: {frame_filename} ({i+1}/{len(sorted_frames)}) ---")

        # 在每次循环开始时，重新初始化对话
        conv = conv_templates[args.conv_mode].copy()

        if hasattr(model, 'is_first_frame'):
            # 如果不使用idea，则每一帧都是“第一帧”
            model.is_first_frame = not use_memory_idea or (i == 0)
        
        # [修改] 构建图片路径：data_root_path / scene_id / frame_filename
        image_path = os.path.join(data_root_path, scene_id, frame_filename)

        if not os.path.exists(image_path):
            print(f"Skipping {frame_filename}, file not found at {image_path}")
            continue

        # [修改] 图片加载与处理 (单张图片)
        try:
            image = Image.open(image_path).convert('RGB')
            # 根据显存情况，可以调整 resize 大小，或者注释掉 resize 使用原始分辨率
            # image = image.resize((800, 450)) 
            
            # process_images 期望输入是 list
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        except Exception as e:
            print(f"Error loading or processing image {frame_filename}: {e}")
            continue

        # [修改] 构建 Prompt (单张图片)
        # 不再需要拼接 CAM_FRONT 等复杂的摄像头前缀，直接使用 <image> token
        if i == 0 or not use_memory_idea:
            # 第一帧，或者不使用记忆时，提供完整 Prompt
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{args.initial_prompt}"
        else:
            # 后续帧，且使用记忆时，只提供图片占位符（或者根据需要添加简单的引导语）
            # 这里的逻辑取决于你的 "Memory Idea" 具体实现。
            # 通常如果是视频流对话，可能只需要图片占位符。
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n" 
            # 如果需要每一帧都重复问题，解开下面这行：
            # prompt = f"{DEFAULT_IMAGE_TOKEN}\n{args.initial_prompt}"

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_for_model = conv.get_prompt()

        # 模型推理
        input_ids = (
            tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .to(model.device)
        )

        split_id = tokenizer.encode('.', add_special_tokens=False)[0]

        inference_start_time = time.time()

        with torch.inference_mode():
            output_ids = model.generate_with_mem_ppl_continue_gen(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size], # 单图 size
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                split_id=split_id,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
            )

        inference_end_time = time.time()

        # 累加统计数据
        frame_inference_time = inference_end_time - inference_start_time
        total_inference_time += frame_inference_time

        generated_ids = output_ids[0]
        num_generated_tokens = len(generated_ids)
        total_generated_tokens += num_generated_tokens

        outputs = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # 更新对话历史 (仅用于显示，实际 context 维护在 KV cache 中)
        conv.messages[-1][-1] = outputs
        
        print(f"Model Output: {outputs}\n")

        print(f"  - Inference time for this frame: {frame_inference_time:.2f}s")
        if frame_inference_time > 0:
            print(f"  - Tokens/sec for this frame: {num_generated_tokens / frame_inference_time:.2f}")
        print("-" * 20)

    total_scene_time = time.time() - total_scene_time_start
    print(f"\n===== Scene {scene_id} Summary (Memory Idea: {use_memory_idea}) =====")
    print(f"Total End-to-End Time: {total_scene_time:.2f}s")
    print(f"Total Inference-Only Time: {total_inference_time:.2f}s")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    if total_inference_time > 0:
        avg_tokens_per_sec = total_generated_tokens / total_inference_time
        print(f"Average Tokens/Second (Inference Only): {avg_tokens_per_sec:.2f}")
    print("=" * 40 + "\n")

    return total_scene_time, total_inference_time, total_generated_tokens


def main():
    # --- 用户需要配置的路径 ---
    # 你的JSON标注文件路径
    json_file_path = "/root/fsas/dataset/BDD-X-Dataset/videos/samples-1k/frames_1fps/frames_info.json" 
    
    # 你的数据根目录。
    # 根据你的描述，图片位于 /root/fsas/dataset/BDD-X-Dataset/videos/samples-1k/frames_1fps/视频名/xx.jpg
    # 所以 data_root_path 应该是:
    data_root_path = "/root/fsas/dataset/BDD-X-Dataset/videos/samples-1k/frames_1fps"
    
    model_path = "/root/fsas/models/LLaVA/llava-v1.6-vicuna-7b"
    model_name = get_model_name_from_path(model_path)

    # 1. 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
    )

    if hasattr(model, 'is_first_frame'):
        model.is_first_frame = True
    else:
        print("Warning: model does not have 'is_first_frame' attribute.")

    model.config.image_aspect_ratio = 'square'

    # 加载阈值表 (如果存在)
    threshold_table_path = "threshold_table.json"
    if os.path.exists(threshold_table_path):
        print(f"Loading existing threshold table from {threshold_table_path}")
        with open(threshold_table_path, 'r') as f:
            threshold_table_json = json.load(f)
        for k, v in threshold_table_json.items():
            token_id = int(k)
            if token_id < tokenizer.vocab_size:
                model.threshold_table[token_id] = v


    # 2. 定义参数
    args = type('Args', (), {
        "conv_mode": "llava_v1",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        # 针对单视图修改了 Prompt
        "initial_prompt": "Suppose you are driving. I am providing you with an image captured by the car's front camera. First, generate a description of the driving scene which includes the key factors for driving planning, including the presence of obstacles and the positions and movements of vehicles and pedestrians and traffic lights. After description, please predict the behavior of ego vehicle, including exactly the driving direction(straight, turn left or turn right) and driving speed(slow, fast or normal)."
    })()

    # 3. 加载JSON文件
    try:
        with open(json_file_path, 'r') as f:
            all_data = json.load(f) # 现在这是一个 list
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    # 遍历JSON中的每个场景
    scene_count = 0
    scenes_to_test = 1

    total_stats = {
        False: {"scenes": 0, "total_time": 0.0, "total_inference_time": 0.0, "total_tokens": 0},
        True: {"scenes": 0, "total_time": 0.0, "total_inference_time": 0.0, "total_tokens": 0},
    }

    # [修改] 遍历逻辑适配 List 结构
    for scene_item in all_data:
        if scene_count >= scenes_to_test:
            print(f"Reached the limit of {scenes_to_test} scenes. Stopping.")
            break
        scene_count += 1

        # [修改] 从字典中获取 video_name 和 saved_frames
        scene_id = scene_item.get("video_name")
        frame_list = scene_item.get("saved_frames")

        if not frame_list:
            print(f"Skipping scene {scene_id}, no frames found.")
            continue

        print(f"\n>>> Start Processing Video Sequence: {scene_id}")

        # --- 模式一：不使用师兄的idea (逐帧独立推理) ---
        if hasattr(model, 'is_first_frame'):
            model.is_first_frame = True 
        # scene_time, inference_time, generated_tokens = run_inference_for_sequence(
        #     args, tokenizer, model, image_processor, context_len, 
        #     scene_id, frame_list, data_root_path, use_memory_idea=False
        # )
        # total_stats[False]["scenes"] += 1
        # total_stats[False]["total_time"] += scene_time
        # total_stats[False]["total_inference_time"] += inference_time
        # total_stats[False]["total_tokens"] += generated_tokens

        # --- 模式二：使用师兄的idea (连续生成) ---
        if hasattr(model, 'is_first_frame'):
            model.is_first_frame = True 
        scene_time, inference_time, generated_tokens = run_inference_for_sequence(
            args, tokenizer, model, image_processor, context_len, 
            scene_id, frame_list, data_root_path, use_memory_idea=True
        )
        total_stats[True]["scenes"] += 1
        total_stats[True]["total_time"] += scene_time
        total_stats[True]["total_inference_time"] += inference_time
        total_stats[True]["total_tokens"] += generated_tokens
    
    print("\n" + "="*25 + " FINAL SUMMARY " + "="*25)
    
    for use_idea, stats in total_stats.items():
        num_scenes = stats["scenes"]
        if num_scenes == 0:
            continue

        avg_time = stats["total_time"] / num_scenes
        avg_inference_time = stats["total_inference_time"] / num_scenes
        
        if stats["total_inference_time"] > 0:
            avg_tps = stats["total_tokens"] / stats["total_inference_time"]
        else:
            avg_tps = 0

        print(f"\n----- Statistics for 'use_memory_idea = {use_idea}' -----")
        print(f"Processed {num_scenes} scenes.")
        print(f"Average End-to-End Time per Scene: {avg_time:.2f}s")
        print(f"Average Inference-Only Time per Scene: {avg_inference_time:.2f}s")
        print(f"Average Tokens/Second (Inference Only, across all scenes): {avg_tps:.2f}")

    print("\n" + "="*67)


if __name__ == "__main__":
    main()