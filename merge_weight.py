import argparse
import os
import sys

import pdb
import json
import torch
from peft import LoraConfig, get_peft_model

from transformers import AutoProcessor

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    # arg_url
    parser.add_argument('--exp_dir', 
        type=str, 
        default="./logs/debug/2025-07-02_20-54-16/")
    return parser.parse_args(args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def main(args):
    args = parse_args(args)
    json_url = os.path.join(args.exp_dir, 'args.json')
    with open(json_url, 'r') as f:
        json_args = json.load(f)
    for key, value in json_args.items():
        setattr(args, key, value)

    args.save_path = args.exp_dir + "/ckpt_model/merged_model"
    args.weight_url = args.exp_dir + "/ckpt_model/pytorch_model.bin"

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    if args.model_id in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
        from model.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        model_id = args.model_id.replace("Qwen/", "")
        model_url = args.model_id
        processor = Qwen2_5_VLProcessor.from_pretrained(
                                                        model_url,
                                                        min_pixels=args.min_visual_tokens *28*28, 
                                                        max_pixels=args.max_visual_tokens *28*28,
                                                        model_max_length=args.model_max_length,
                                                        )

        from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_url,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            # quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
        )
        
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=["visual"])
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    state_dict = torch.load(args.weight_url, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict, safe_serialization=False)
    processor.save_pretrained(args.save_path)

if __name__ == "__main__":
    main(sys.argv[1:])