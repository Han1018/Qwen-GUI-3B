import os
import re
import ast
import sys
import pdb
import json
import torch
import wandb
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.distributed as dist
from accelerate.utils import gather_object
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_json
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
logging.basicConfig(level=logging.INFO)

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels=28*28*256, max_pixels=14*14*4*1280):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_point_from_qwen25vl_format(point, orig_height, orig_width, factor=28, min_pixels=28*28*256, max_pixels=14*14*4*1280):
    # 計算 resize 過後的尺寸
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)

    # 縮放比例
    scale_w = orig_width / new_width
    scale_h = orig_height / new_height

    x_resized, y_resized = point
    x_orig = round(x_resized * scale_w)
    y_orig = round(y_resized * scale_h)

    # Clamp 到圖片範圍
    x_orig = max(0, min(x_orig, orig_width - 1))
    y_orig = max(0, min(y_orig, orig_height - 1))

    return [x_orig, y_orig]

def extract_coordinates_str(text):
    # 正規表達式：找到中括號內的內容（可以包含小數點）
    match = re.search(r'\[\s*[\d\.]+\s*,\s*[\d\.]+\s*\]', text)
    if match:
        return match.group(0)  # 直接回傳整個匹配到的字串
    else:
        return None

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def get_bbox(bbox, img_size, xy_int):
    x1, y1, w, h = bbox
    weight, height = img_size

    # x1y1wh to x1y1x2y2
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox

def pointinbbox(pred_point, gt_bbox):
    # pred_point: [x, y] in [0, 1]
    # gt_bbox: [x1, y1, x2, y2] in [0, 1]
    if (gt_bbox[0] <= pred_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_point[1] <= gt_bbox[3]):
        return True
    else:
        return False

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0], point[1]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def calculate_screenspot_metrics(results):
    metrics = {}
    for type in results:
        num_step = 0
        num_success = 0

        for step in results[type]:
            num_step += 1
            num_success += step["acc"]

        metrics[f"{type} Success Rate"] = num_success / num_step

    for key, value in metrics.items():
        logging.info(f"[{key}]: {value}")
    return metrics

@torch.no_grad()
def validate_screenspot(val_loader, model_engine, processor, epoch, global_step, writer, args, media=True):
    model_engine.eval()

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_think_grounding = args.think_grounding

    metric = 0
    for i, input_dict in enumerate(tqdm(val_loader)):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        with torch.no_grad():
            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                )

        forward_dict.update(image_grid_thw=input_dict["image_sizes"]) if "image_sizes" in input_dict else None
        forward_dict.update(patch_assign=input_dict["patch_assign"]) if "patch_assign" in input_dict else None
        forward_dict.update(patch_assign_len=input_dict["patch_assign_len"]) if "patch_assign_len" in input_dict else None
        forward_dict.update(patch_pos=input_dict["patch_pos"]) if "patch_pos" in input_dict else None
        forward_dict.update(select_mask=input_dict["select_mask"]) if "select_mask" in input_dict else None
        
        try:
            generate_ids = model_engine.generate(**forward_dict, 
                                    max_new_tokens=2048, 
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    )
            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            meta = input_dict['meta_data'][0]

            # print(f"generated_texts: {generated_texts}")
            # print(f"answer: {meta['bbox']}")
            if is_think_grounding:
                generated_texts_ori = generated_texts
                generated_texts = extract_coordinates_str(generated_texts)
                if generated_texts is None:
                    print(f"generated_texts: {generated_texts}")
                    raise ValueError("No coordinates found in the generated text.")
            
            pred_point = ast.literal_eval(generated_texts)
            
            if len(pred_point) == 4:
                pred_point = [(pred_point[0]+pred_point[2])/2, (pred_point[1]+pred_point[3])/2]
            if args.model_id =='Qwen/Qwen2.5-VL-3B-Instruct':
                pred_point = convert_point_from_qwen25vl_format(pred_point, meta['img_size'][1], meta['img_size'][0], min_pixels=args.min_visual_tokens*28*28, max_pixels=args.max_visual_tokens*28*28)
                generated_texts = str(pred_point)

        except Exception as e:
            print(f"An error occurred: {e}")

        outputs = {"split": meta['split'], 'data_type': meta['data_type'],
                    "anno_id": meta['id'], "img_path": meta['img_url_abs'], "instruction": meta['task'], "sentence": generated_texts,
                    "bbox": meta['bbox'], 
                    "explanation": generated_texts_ori if is_think_grounding else None,
                    "meta": meta}

        generated_texts_unique.append(generated_texts)
        answers_unique.append(meta['bbox'])
        outputs_unique.append(outputs)

    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    if global_rank == 0:
        results = {}
        for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
            anno_id = output_i['anno_id']
            split_i = output_i['split']
            if split_i not in results:
                results[split_i] = {}

            type_i = output_i['data_type']
            if type_i not in results[split_i]:
                results[split_i][type_i] = []

            step_result = output_i.copy()

            img_size = output_i['meta']['img_size']
            gt_bbox = get_bbox(ans_i, img_size, args.xy_int)
            step_result['gt_bbox'] = gt_bbox

            try:
                pred_point = ast.literal_eval(pred_i)
                step_result['pred_point'] = pred_point

                if pointinbbox(pred_point, gt_bbox):
                    step_result["acc"] = 1
                else:
                    step_result["acc"] = 0
                    
            except Exception as e:
                print(e)
                print(f"format wrong with {anno_id}'s prediction: {pred_i}")
                step_result["acc"] = 0

            results[split_i][type_i].append(step_result)

        eval_dict = {}
        for split in results.keys():
            logging.info("==="*10)
            logging.info(f"{split}")
            logging.info("==="*10)
            eval_dict[split] = calculate_screenspot_metrics(results[split])

        if not args.debug:
            for split in eval_dict.keys():
                for key, value in eval_dict[split].items():
                    if isinstance(value, list):
                        continue
                    writer.add_scalar(f"metrics/screenspot/{split}/{key}", value, epoch)
                    wandb.log({f"metrics/screenspot/{split}/{key}": value}, step=global_step)

        score_all = [value for split in eval_dict.values() for value in split.values()]
        metric = sum(score_all) / len(score_all)
        eval_dict['Avg Success Rate'] = metric
        writer.add_scalar("metrics/screenspot/Avg Success Rate", metric, epoch)
        wandb.log({"metrics/screenspot/Avg Success Rate": metric}, step=global_step)

        if media:
            image_save_dir = os.path.join(args.tmp_dir, "images")
            os.makedirs(image_save_dir, exist_ok=True)
            images_list = []
            random.seed(123)  # Set the random seed to ensure reproducibility
            for split in results.keys():
                for type in results[split].keys():
                    samples = random.sample(results[split][type], 10)  # Sample 10 examples
                    for sample in samples:
                    # for sample in results[split][type]:
                        try:
                            img_anno = sample['anno_id']
                            img_url = sample['img_path']
                            img_inst = sample['instruction']
                            gt_bbox = sample['gt_bbox']
                            if 'pred_point' in sample:
                                pred_point = sample['pred_point']
                                img_array = draw_point_bbox(img_url, pred_point, gt_bbox, radius=5, line=3)
                            else:
                                img_array = draw_point_bbox(img_url, None, gt_bbox)
                            img = Image.fromarray(img_array)
                            img.save(os.path.join(image_save_dir, f"example_{epoch}_{split}_{type}_{img_anno}.png"))
                            # images = wandb.Image(img_array, caption=f"{split}/{type}/{img_anno}_{img_inst}")
                            # images_list.append(images)
                        except Exception as e:
                            pdb.set_trace()
                            print(f"An error occurred: {e}")
            # wandb.log({"examples": images_list}, step=global_step)
 
        save_json(results, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_tmp_dict.json'))
        save_json(eval_dict, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_res_dict.json'))

    metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric