import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

DEFAULT_PROMPT = """You are an autonomous driving assistant. I'm providing you with 6 images captured simultaneously by the car's cameras (front, front-left, front-right, back, back-left, back-right).

Your task is to synthesize all 6 views into a SINGLE holistic understanding of the environment, and predict the ego vehicle's behavior.

Output EXACTLY ONE JSON object with exactly these keys: \"description\", \"direction\", \"speed\".
- \"description\": a short scene summary.
- \"direction\": one of [\"straight\", \"turn left\", \"turn right\"].
- \"speed\": one of [\"slow\", \"normal\", \"fast\"].

Output only valid JSON."""


def parse_ground_truth(answer_string: str, parse_speed: bool = True, parse_direction: bool = True) -> str:
    answer_string = answer_string.lower()

    direction_map = {
        "straight": ["straight"],
        "left": ["left"],
        "right": ["right"],
    }
    speed_map = {
        "slow": ["slow", "not moving", "stopped"],
        "normal": ["normal", "moderate"],
        "fast": ["fast"],
    }

    gt_direction = "unknown"
    gt_speed = "unknown"

    for key, values in direction_map.items():
        if any(value in answer_string for value in values):
            gt_direction = key
            break

    for key, values in speed_map.items():
        if any(value in answer_string for value in values):
            gt_speed = key
            break

    if parse_direction:
        return gt_direction
    if parse_speed:
        return gt_speed
    return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)


def build_qa_prompt(camera_names: List[str], instruction: str, question: str, scene_description: str = "") -> str:
    image_prompt_parts = [f"- {name}: {DEFAULT_IMAGE_TOKEN}" for name in camera_names]
    image_section = "The following images are provided from six camera angles:\n" + "\n".join(image_prompt_parts)
    if scene_description.strip():
        return (
            f"{image_section}\n\n{instruction.strip()}\n\n"
            f"Scene context: {scene_description.strip()}\n\nQuestion: {question.strip()}"
        )
    return f"{image_section}\n\n{instruction.strip()}\n\nQuestion: {question.strip()}"


def build_image_paths(image_paths_dict: Dict[str, str], data_root_path: str) -> Tuple[List[str], List[str]]:
    image_files: List[str] = []
    camera_names: List[str] = []
    for camera_name in CAMERA_ORDER:
        if camera_name not in image_paths_dict:
            continue
        rel_path = image_paths_dict[camera_name]
        image_files.append(os.path.join(data_root_path, camera_name, os.path.basename(rel_path)))
        camera_names.append(camera_name)
    return image_files, camera_names


def save_selected_scenes(scene_ids: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    selected_scene_path = os.path.join(output_dir, "selected_scenes.txt")
    with open(selected_scene_path, "w", encoding="utf-8") as file:
        for scene_id in scene_ids:
            file.write(f"{scene_id}\n")
    print(f"selected scenes saved to: {selected_scene_path}")


def load_drivelm_samples(
    data_path: str,
    image_root: str,
    num_scenes: int,
    scene_seed: int,
    qa_groups: List[str],
    output_dir: str,
    instruction: str,
) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as file:
        all_data = json.load(file)

    scene_ids = list(all_data.keys())
    if num_scenes > 0 and num_scenes < len(scene_ids):
        rng = random.Random(scene_seed)
        selected_scene_ids = sorted(rng.sample(scene_ids, num_scenes))
    else:
        selected_scene_ids = sorted(scene_ids)

    print("selected scenes for finetune:")
    for scene_id in selected_scene_ids:
        print(scene_id)
    save_selected_scenes(selected_scene_ids, output_dir)

    samples: List[Dict[str, Any]] = []
    for scene_id in selected_scene_ids:
        scene_data = all_data[scene_id]
        key_frames = scene_data.get("key_frames", {})
        scene_description = scene_data.get("scene_description", "")
        for frame_id, frame_info in key_frames.items():
            qa_info = frame_info.get("QA", {})
            image_paths_dict = frame_info.get("image_paths", {})
            if not image_paths_dict:
                continue

            image_files, camera_names = build_image_paths(image_paths_dict, image_root)
            if not image_files or not all(os.path.exists(path) for path in image_files):
                continue

            for qa_group in qa_groups:
                qa_items = qa_info.get(qa_group, [])
                if not isinstance(qa_items, list):
                    continue
                for qa_item in qa_items:
                    question = qa_item.get("Q", "")
                    answer = qa_item.get("A", "")
                    if not isinstance(question, str) or not isinstance(answer, str):
                        continue
                    question = question.strip()
                    answer = answer.strip()
                    if not question or not answer:
                        continue

                    samples.append(
                        {
                            "scene_id": scene_id,
                            "frame_id": frame_id,
                            "qa_group": qa_group,
                            "prompt": build_qa_prompt(camera_names, instruction, question, scene_description),
                            "target": answer,
                            "image_files": image_files,
                        }
                    )
                    print(
                        f"built sample: scene={scene_id} frame={frame_id} "
                        f"group={qa_group} question={json.dumps(question, ensure_ascii=False)} "
                        f"target={json.dumps(answer, ensure_ascii=False)}"
                    )
    return samples


def build_text_tensors(
    tokenizer: Any,
    conv_mode: str,
    prompt: str,
    target: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_conv = conv_templates[conv_mode].copy()
    prompt_conv.append_message(prompt_conv.roles[0], prompt)
    prompt_conv.append_message(prompt_conv.roles[1], None)
    prompt_text = prompt_conv.get_prompt()

    full_conv = conv_templates[conv_mode].copy()
    full_conv.append_message(full_conv.roles[0], prompt)
    full_conv.append_message(full_conv.roles[1], target)
    full_text = full_conv.get_prompt()

    prompt_ids = ensure_tensor(tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt"))
    input_ids = ensure_tensor(tokenizer_image_token(full_text, tokenizer, return_tensors="pt"))
    labels = input_ids.clone()
    labels[: prompt_ids.shape[0]] = IGNORE_INDEX

    return input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)


def load_image_tensor(image_files: List[str], image_processor: Any, model: Any) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    images = [Image.open(path).convert("RGB").resize((800, 450)) for path in image_files]
    image_sizes = [image.size for image in images]
    image_tensor = process_images(images, image_processor, model.config)
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    return image_tensor, image_sizes


def save_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("这个最小脚本默认要求 CUDA 环境。")

    set_seed(args.seed)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
    )

    model.train()
    model.config.use_cache = False
    model.config.image_aspect_ratio = args.image_aspect_ratio

    samples = load_drivelm_samples(
        data_path=args.data_path,
        image_root=args.image_root,
        num_scenes=args.num_scenes,
        scene_seed=args.scene_seed,
        qa_groups=args.qa_groups,
        output_dir=args.output_dir,
        instruction=args.prompt,
    )
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("没有加载到可训练样本，请检查 data_path 和 image_root。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(samples)
        running_loss = 0.0

        for sample in samples:
            input_ids, labels = build_text_tensors(
                tokenizer=tokenizer,
                conv_mode=args.conv_mode,
                prompt=sample["prompt"],
                target=sample["target"],
                device=model.device,
            )
            image_tensor, image_sizes = load_image_tensor(sample["image_files"], image_processor, model)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                images=image_tensor,
                image_sizes=image_sizes,
                return_dict=True,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * args.grad_accum_steps

            global_step += 1
            if global_step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"scene={sample['scene_id']} frame={sample['frame_id']} loss={running_loss / args.log_every:.4f}"
                )
                running_loss = 0.0

        if global_step % args.grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        save_model(model, tokenizer, epoch_dir)
        print(f"saved: {epoch_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    save_model(model, tokenizer, final_dir)
    print(f"final model saved to: {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DriveLM finetuning script for LLaVA")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--image-aspect-ratio", type=str, default="square")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--qa-groups", nargs="+", default=["perception", "behavior"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
