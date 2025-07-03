import re
import torch

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head", "vision_model", "img_projection", "visual_model"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    # lora_namespan_exclude += ["vision_model", "img_projection", "visual_model"]
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

def parse_layer_type(str_ranges, L, default=0):
    # 0 is without layer token selection, 1 is with layer token selection
    result = [default] * L
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str_ranges)
    for start, end, value in matches:
        start, end, value = int(start) - 1, int(end) - 1, int(value)
        if end >= L:
            end = L - 1
        result[start:end + 1] = [value] * (end - start + 1)
    return result