import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM


def merge_lora(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    lora_cfg_pretrained = LlavaConfig.from_pretrained(args.model_path)
    if hasattr(lora_cfg_pretrained, 'quantization_config'):
        lora_cfg_pretrained.quantization_config = {}

    print('Loading LLaVA base model...')
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_base,
        low_cpu_mem_usage=True,
        config=lora_cfg_pretrained,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    print(f'Loaded base model on {model_device}, dtype={model_dtype}, 4bit={getattr(model, "is_loaded_in_4bit", False)}')

    token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty(token_num, token_dim, device=model_device, dtype=model_dtype)
        )
        model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(token_num, token_dim, device=model_device, dtype=model_dtype)
        )

    print('Loading additional LLaVA weights...')
    non_lora_path = os.path.join(args.model_path, 'non_lora_trainables.bin')
    non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
    non_lora_trainables = {
        (key[11:] if key.startswith('base_model.') else key): value
        for key, value in non_lora_trainables.items()
    }
    if any(key.startswith('model.model.') for key in non_lora_trainables):
        non_lora_trainables = {
            (key[6:] if key.startswith('model.') else key): value
            for key, value in non_lora_trainables.items()
        }
    model.load_state_dict(non_lora_trainables, strict=False, assign=True)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, args.model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    if hasattr(model.config, 'quantization_config'):
        model.config.quantization_config = {}
    print('Saving merged model...')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
