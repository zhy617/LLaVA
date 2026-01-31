import os
import json
import time

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

def run_inference_for_sequence(
    args, 
    tokenizer, 
    model, 
    image_processor, 
    context_len, 
    scene_id: str, 
    key_frames_data: Dict, 
    data_root_path: str,
    use_memory_idea: bool = False
):
    """
    对一个数据序列（一个场景的所有关键帧）进行连续推理
    """
    print(f"===== Processing Scene: {scene_id} (Memory Idea: {use_memory_idea}) =====")

    total_scene_time_start = time.time()
    total_generated_tokens = 0
    total_inference_time = 0

    # 获取并排序关键帧ID，以确保按顺序处理
    # 注意：字典在Python 3.7+中保持插入顺序，但排序更保险
    key_frame_ids: List[Dict] = sorted(key_frames_data.keys())

    last_output = None

    # 3. 根据是否使用idea，决定conv对象的位置
    
    for i, frame_id in enumerate(key_frame_ids):
        print(f"--- Processing Key Frame: {frame_id} ---")

        # 在每次循环开始时，重新初始化对话，避免历史累积
        conv = conv_templates[args.conv_mode].copy()

        if hasattr(model, 'is_first_frame'):
            # 如果不使用idea，则每一帧都是“第一帧”
            model.is_first_frame = not use_memory_idea or (i == 0)
        
        frame_info: Dict[str, Dict] = key_frames_data[frame_id]
        assert frame_info is not None, f"Frame info for {frame_id} is None"
        image_paths_dict= frame_info.get("image_paths")

        if not image_paths_dict:
            print(f"Skipping {frame_id}, 'image_paths' not found.")
            continue

        # 1. 准备多张图片输入，并确保顺序固定
        # 定义一个固定的摄像头顺序，以保证 prompt 和 image_tensor 的顺序一致
        camera_order = [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
        ]
        
        image_files = []
        camera_names = []
        for camera_view in camera_order:
            if camera_view in image_paths_dict:
                rel_path = image_paths_dict[camera_view]
                image_files.append(os.path.join(data_root_path, camera_view, os.path.basename(rel_path)))
                camera_names.append(camera_view)

        # 这里我们只使用前置摄像头的图片进行测试
        # image_files = []
        # front_camera_view = "CAM_FRONT"
        # if front_camera_view in image_paths_dict:
        #     rel_path = image_paths_dict[front_camera_view]
        #     image_files.append(os.path.join(data_root_path, front_camera_view, os.path.basename(rel_path)))
        #     print(f"--- Info: Testing with a single image ({front_camera_view}) ---")
        # else:
        #     # 如果找不到前置摄像头，就用任意第一张图作为备选
        #     if image_paths_dict:
        #         first_camera_view, rel_path = next(iter(image_paths_dict.items()))
        #         image_files.append(os.path.join(data_root_path, first_camera_view, os.path.basename(rel_path)))
        #         print(f"--- Info: CAM_FRONT not found. Testing with a single image (fallback: {first_camera_view}) ---")


        # 检查图片是否存在
        if not all(os.path.exists(f) for f in image_files):
            print(f"Skipping {frame_id}, not all images found.")
            # 打印找不到的文件以供调试
            for f in image_files:
                if not os.path.exists(f):
                    print(f"  File not found: {f}")
            continue

        try:
            images = [Image.open(f).convert('RGB').resize((800, 450)) for f in image_files]
            image_tensor = process_images(images, image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        except Exception as e:
            print(f"Error loading or processing images for {frame_id}: {e}")
            continue

        # 关键修改：构建带有摄像头方位信息的 prompt
        image_prompt_parts = [f"- {name}: {DEFAULT_IMAGE_TOKEN}" for name in camera_names]
        image_section = "The following images are provided from different camera angles:\n" + "\n".join(image_prompt_parts)
        
        prompt = f"{image_section}\n\n{args.initial_prompt}"

        # if i == 0: # 或者 if model.is_first_frame:
        #     # 第一帧：构建包含完整指令的 prompt
        #     prompt = f"{DEFAULT_IMAGE_TOKEN * len(images)}\n" + args.initial_prompt
        # else:
        #     # 后续帧：只提供图片占位符，让模型自己利用记忆续写
        #     prompt = f"{DEFAULT_IMAGE_TOKEN * len(images)}"

        # prompt = "Suppose you are driving, and I'm providing you with six images captured by the car's front, front-left, front-right, back, back-left and back-right camera. First, generate a description of the driving scene which includes the key factors for driving planning, including the presence of obstacles and the positions and movements of vehicles and pedestrians and traffic lights. After description, please predict the behavior of ego vehicle, including exactly the driving direction(straight, turn left or turn right) and driving speed(slow, fast or normal)."

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_for_model = conv.get_prompt()

        # 3. 模型推理
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
                image_sizes=[img.size for img in images],
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
        # outputs = tokenizer.decode(output_ids[input_ids.shape[1]:]).strip()
        
        # 4. 更新对话历史
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

    threshold_table_path = "threshold_table.json"
    if os.path.exists(threshold_table_path):
        print(f"Loading existing threshold table from {threshold_table_path}")
        with open(threshold_table_path, 'r') as f:
            threshold_table_json = json.load(f)
        # from json to tensor
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

    # 遍历JSON中的每个场景

    scene_count = 0
    scenes_to_test = 5

    for scene_id, scene_data in all_data.items():
        if scene_count >= scenes_to_test:
            print(f"Reached the limit of {scenes_to_test} scenes. Stopping.")
            break
        scene_count += 1

        key_frames = scene_data.get("key_frames")
        if not key_frames:
            print(f"Skipping scene {scene_id}, no 'key_frames' found.")
            continue

        # --- 模式一：不使用师兄的idea (逐帧独立推理) ---
        if hasattr(model, 'is_first_frame'):
            model.is_first_frame = True # 确保从干净的状态开始
        # run_inference_for_sequence(args, tokenizer, model, image_processor, context_len, scene_id, key_frames, data_root_path, use_memory_idea=False)

        # --- 模式二：使用师兄的idea (连续生成) ---
        if hasattr(model, 'is_first_frame'):
            model.is_first_frame = True # 同样确保从干净的状态开始
        run_inference_for_sequence(args, tokenizer, model, image_processor, context_len, scene_id, key_frames, data_root_path, use_memory_idea=True)


if __name__ == "__main__":
    main()