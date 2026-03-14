import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llava.constants import DEFAULT_IMAGE_TOKEN

CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

DEFAULT_INSTRUCTION = (
    "You are an autonomous driving assistant. "
    "Given a stitched six-camera surround-view image, answer the question accurately."
)


@dataclass
class FrameRecord:
    scene_id: str
    frame_id: str
    image_relpath: str


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_image_path(image_root: Path, camera_name: str, raw_path: str) -> Optional[Path]:
    raw = Path(raw_path)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    candidates.append(image_root / raw_path)
    candidates.append(image_root / camera_name / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def open_and_resize_image(image_path: Path, tile_size: Tuple[int, int]) -> Any:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if hasattr(Image, "Resampling"):
        return image.resize(tile_size, Image.Resampling.BICUBIC)
    return image.resize(tile_size, 3)


def make_surround_view_image(
    image_root: Path,
    image_paths_dict: Dict[str, str],
    tile_size: Tuple[int, int],
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Optional[Any]:
    from PIL import Image

    tile_w, tile_h = tile_size
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2), background_color)

    found_any = False
    for index, camera_name in enumerate(CAMERA_ORDER):
        if camera_name not in image_paths_dict:
            continue

        resolved = resolve_image_path(image_root, camera_name, image_paths_dict[camera_name])
        if resolved is None:
            continue

        found_any = True
        row = 0 if index < 3 else 1
        col = index % 3
        tile = open_and_resize_image(resolved, tile_size)
        canvas.paste(tile, (col * tile_w, row * tile_h))

    if not found_any:
        return None
    return canvas


def build_human_prompt(
    question: str,
    instruction: str,
    scene_description: str,
    include_scene_description: bool,
) -> str:
    lines = [DEFAULT_IMAGE_TOKEN, instruction.strip()]
    if include_scene_description and scene_description.strip():
        lines.append(f"Scene context: {scene_description.strip()}")
    lines.append(f"Question: {question.strip()}")
    return "\n\n".join(lines)


def sanitize_text(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def save_stitched_frame(
    output_dir: Path,
    scene_id: str,
    frame_id: str,
    image: Any,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{scene_id}__{frame_id}.jpg"
    save_path = output_dir / filename
    image.save(save_path, format="JPEG", quality=95)
    return filename


def parse_frame_key_from_image_relpath(image_relpath: str) -> Optional[Tuple[str, str]]:
    stem = Path(image_relpath).stem
    if "__" not in stem:
        return None
    scene_id, frame_id = stem.split("__", 1)
    if not scene_id or not frame_id:
        return None
    return scene_id, frame_id


def load_existing_records(output_json_path: Path) -> List[Dict]:
    if not output_json_path.exists():
        return []
    with open(output_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise RuntimeError("Existing output JSON is not a list. Please rebuild conversion output.")
    return data


def collect_existing_conversion_stats(output_json_path: Path, stitched_image_dir: Path) -> Dict[str, int]:
    records = load_existing_records(output_json_path)
    unique_images = set()
    for record in records:
        image_relpath = record.get("image", "") if isinstance(record, dict) else ""
        if isinstance(image_relpath, str) and image_relpath:
            unique_images.add(image_relpath)

    existing_stitched = 0
    for image_relpath in unique_images:
        if (stitched_image_dir / image_relpath).exists():
            existing_stitched += 1

    return {
        "num_frames": existing_stitched,
        "num_samples": len(records),
    }


def select_scene_ids(all_data: Dict[str, Dict], num_scenes: int, scene_seed: int) -> List[str]:
    scene_ids = list(all_data.keys())
    if num_scenes > 0 and num_scenes < len(scene_ids):
        rng = random.Random(scene_seed)
        return sorted(rng.sample(scene_ids, num_scenes))
    return sorted(scene_ids)


def build_llava_training_json(
    drivelm_json_path: Path,
    image_root: Path,
    stitched_image_dir: Path,
    output_json_path: Path,
    qa_groups: Sequence[str],
    instruction: str,
    include_scene_description: bool,
    tile_size: Tuple[int, int],
    num_scenes: int,
    scene_seed: int,
    max_samples: int,
    conversion_mode: str,
) -> Dict[str, int]:
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency: Pillow. Please install it first, e.g. `pip install pillow`."
        ) from error

    try:
        with open(drivelm_json_path, "r", encoding="utf-8") as file:
            all_data = json.load(file)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Invalid JSON in --drivelm-json. "
            f"line={error.lineno}, col={error.colno}, msg={error.msg}. "
            "Please ensure the annotation file is strict JSON (no trailing commas/comments)."
        ) from error

    selected_scene_ids = select_scene_ids(all_data, num_scenes=num_scenes, scene_seed=scene_seed)

    existing_records: List[Dict] = []
    existing_sample_ids = set()
    frame_cache: Dict[Tuple[str, str], FrameRecord] = {}

    if conversion_mode == "resume":
        existing_records = load_existing_records(output_json_path)
        for record in existing_records:
            if not isinstance(record, dict):
                continue
            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id:
                existing_sample_ids.add(sample_id)

            image_relpath = record.get("image", "")
            if not isinstance(image_relpath, str) or not image_relpath:
                continue
            frame_key = parse_frame_key_from_image_relpath(image_relpath)
            if frame_key is None:
                continue
            frame_cache[frame_key] = FrameRecord(
                scene_id=frame_key[0],
                frame_id=frame_key[1],
                image_relpath=image_relpath,
            )

    records: List[Dict] = list(existing_records)
    new_frames = 0
    new_samples = 0

    if max_samples > 0 and len(records) >= max_samples:
        records = records[:max_samples]
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False)
        return {
            "num_scenes": len(selected_scene_ids),
            "num_frames": len(frame_cache),
            "num_samples": len(records),
            "num_new_frames": 0,
            "num_new_samples": 0,
        }

    for scene_id in selected_scene_ids:
        scene_data = all_data.get(scene_id, {})
        scene_description = scene_data.get("scene_description", "")
        key_frames = scene_data.get("key_frames", {})

        for frame_id, frame_info in key_frames.items():
            image_paths_dict = frame_info.get("image_paths", {})
            if not isinstance(image_paths_dict, dict) or not image_paths_dict:
                continue

            frame_key = (scene_id, frame_id)
            if frame_key not in frame_cache:
                output_filename = f"{scene_id}__{frame_id}.jpg"
                output_path = stitched_image_dir / output_filename

                if conversion_mode == "resume" and output_path.exists():
                    frame_cache[frame_key] = FrameRecord(
                        scene_id=scene_id,
                        frame_id=frame_id,
                        image_relpath=output_filename,
                    )
                else:
                    stitched = make_surround_view_image(
                        image_root=image_root,
                        image_paths_dict=image_paths_dict,
                        tile_size=tile_size,
                    )
                    if stitched is None:
                        continue
                    image_relpath = save_stitched_frame(
                        output_dir=stitched_image_dir,
                        scene_id=scene_id,
                        frame_id=frame_id,
                        image=stitched,
                    )
                    frame_cache[frame_key] = FrameRecord(
                        scene_id=scene_id,
                        frame_id=frame_id,
                        image_relpath=image_relpath,
                    )
                    new_frames += 1

            qa_info = frame_info.get("QA", {})
            if not isinstance(qa_info, dict):
                continue

            for qa_group in qa_groups:
                qa_items = qa_info.get(qa_group, [])
                if not isinstance(qa_items, list):
                    continue

                for idx, qa_item in enumerate(qa_items):
                    question = qa_item.get("Q", "") if isinstance(qa_item, dict) else ""
                    answer = qa_item.get("A", "") if isinstance(qa_item, dict) else ""
                    if not isinstance(question, str) or not isinstance(answer, str):
                        continue

                    question = question.strip()
                    answer = answer.strip()
                    if not question or not answer:
                        continue

                    sample_id = f"{scene_id}_{frame_id}_{qa_group}_{idx}"
                    if sample_id in existing_sample_ids:
                        continue

                    human_value = build_human_prompt(
                        question=question,
                        instruction=instruction,
                        scene_description=scene_description,
                        include_scene_description=include_scene_description,
                    )

                    records.append(
                        {
                            "id": sample_id,
                            "image": frame_cache[frame_key].image_relpath,
                            "conversations": [
                                {"from": "human", "value": human_value},
                                {"from": "gpt", "value": sanitize_text(answer)},
                            ],
                        }
                    )
                    existing_sample_ids.add(sample_id)
                    new_samples += 1

                    if max_samples > 0 and len(records) >= max_samples:
                        break
                if max_samples > 0 and len(records) >= max_samples:
                    break
            if max_samples > 0 and len(records) >= max_samples:
                break
        if max_samples > 0 and len(records) >= max_samples:
            break

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)

    return {
        "num_scenes": len(selected_scene_ids),
        "num_frames": len(frame_cache),
        "num_samples": len(records),
        "num_new_frames": new_frames,
        "num_new_samples": new_samples,
    }


def run_official_qlora_train(args: argparse.Namespace) -> None:
    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required when --run-train true")

    cmd = [
        args.deepspeed_bin,
        "llava/train/train_mem.py",
        "--deepspeed",
        args.deepspeed_config,
        "--lora_enable",
        "True",
        "--bits",
        "4",
        "--model_name_or_path",
        args.model_name_or_path,
        "--version",
        args.prompt_version,
        "--data_path",
        args.output_json,
        "--image_folder",
        args.stitched_image_dir,
        "--vision_tower",
        args.vision_tower,
        "--mm_vision_select_layer",
        str(args.mm_vision_select_layer),
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--bf16",
        "True",
        "--output_dir",
        args.train_output_dir,
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--logging_steps",
        str(args.logging_steps),
        "--tf32",
        "True",
        "--model_max_length",
        str(args.model_max_length),
        "--gradient_checkpointing",
        "True",
        "--lazy_preprocess",
        "True",
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--report_to",
        args.report_to,
    ]

    # For merged checkpoints (e.g., LLaVA-1.5/1.6), mm_projector is already included.
    # Keep this arg optional for compatibility with the old two-stage recipe.
    if args.pretrain_mm_mlp_adapter:
        cmd.extend(["--pretrain_mm_mlp_adapter", args.pretrain_mm_mlp_adapter])

    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    print("\nLaunching training command:\n")
    print(" ".join(cmd))
    print("")

    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DriveLM data to official LLaVA format, then optionally run official QLoRA training."
    )

    parser.add_argument("--drivelm-json", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--work-dir", type=str, required=True)

    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--stitched-image-dir", type=str, default="")

    parser.add_argument("--qa-groups", nargs="+", default=["perception", "behavior"])
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--include-scene-description", type=str2bool, default=True)
    parser.add_argument("--tile-width", type=int, default=448)
    parser.add_argument("--tile-height", type=int, default=252)
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--conversion-mode",
        type=str,
        choices=["rebuild", "resume", "skip"],
        default="resume",
        help="rebuild: regenerate conversion from scratch; resume: append missing samples; skip: do not convert and use existing files.",
    )

    parser.add_argument("--run-train", type=str2bool, default=False)

    parser.add_argument("--deepspeed-bin", type=str, default="deepspeed")
    parser.add_argument("--deepspeed-config", type=str, default="./scripts/zero2.json")
    parser.add_argument("--model-name-or-path", type=str, default="")
    parser.add_argument("--prompt-version", type=str, default="v1")
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--pretrain-mm-mlp-adapter", type=str, default="")
    parser.add_argument("--mm-vision-select-layer", type=int, default=-2)

    parser.add_argument("--train-output-dir", type=str, default="")
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--hf-endpoint", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir)
    output_json = Path(args.output_json) if args.output_json else work_dir / "drivelm_llava_train.json"
    stitched_image_dir = (
        Path(args.stitched_image_dir) if args.stitched_image_dir else work_dir / "stitched_images"
    )

    args.output_json = str(output_json)
    args.stitched_image_dir = str(stitched_image_dir)

    if not args.train_output_dir:
        args.train_output_dir = str(work_dir / "checkpoints")

    stats: Dict[str, int]
    if args.conversion_mode == "skip":
        if not output_json.exists():
            raise RuntimeError(f"conversion-mode=skip but output JSON not found: {output_json}")
        if not stitched_image_dir.exists():
            raise RuntimeError(f"conversion-mode=skip but stitched image dir not found: {stitched_image_dir}")
        stats = collect_existing_conversion_stats(output_json, stitched_image_dir)
        print("DriveLM -> LLaVA conversion skipped. Reusing existing files.")
    else:
        if args.conversion_mode == "rebuild":
            if output_json.exists():
                output_json.unlink()
            stitched_image_dir.mkdir(parents=True, exist_ok=True)

        stats = build_llava_training_json(
            drivelm_json_path=Path(args.drivelm_json),
            image_root=Path(args.image_root),
            stitched_image_dir=stitched_image_dir,
            output_json_path=output_json,
            qa_groups=args.qa_groups,
            instruction=args.instruction,
            include_scene_description=args.include_scene_description,
            tile_size=(args.tile_width, args.tile_height),
            num_scenes=args.num_scenes,
            scene_seed=args.scene_seed,
            max_samples=args.max_samples,
            conversion_mode=args.conversion_mode,
        )
        print("DriveLM -> LLaVA conversion done.")

    print(
        json.dumps(
            {
                "conversion_mode": args.conversion_mode,
                "output_json": args.output_json,
                "stitched_image_dir": args.stitched_image_dir,
                **stats,
            },
            indent=2,
        )
    )

    if args.run_train:
        if not output_json.exists():
            raise RuntimeError(f"Training requires converted JSON file: {output_json}")
        if not stitched_image_dir.exists():
            raise RuntimeError(f"Training requires stitched image dir: {stitched_image_dir}")
        run_official_qlora_train(args)


if __name__ == "__main__":
    main()
