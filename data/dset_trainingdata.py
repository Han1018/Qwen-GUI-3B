import os
import cv2
import pdb
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from IPython.display import display

import sys
sys.path.append('.')
from data.template import screenspot_to_qwen, batch_add_answer
dataset_mapping = {
    "showui-desktop": "ShowUI-desktop",
    "showui-web": "ShowUI-web-8k",
    "amex": "AMEX-8k",
    "uground": "UGround-V1-8k",
    "training-eval": "Training-data"
}

def log_message(message):
    log_file_path = "log.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{message}\n")
        
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset,
        json_data,
        processor,
        inference=False,
        args_dict={},
    ):
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.inference = inference

        # pdb.set_trace()
        self.base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get('samples_per_epoch', 1)
        self.xy_int = args_dict.get('xy_int', False)
        self.think_grounding = args_dict.get('think_grounding', False)

        print(f"Dataset: {dataset}; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def smart_resize(self, height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
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

    def convert_bbox_to_qwen25vl_format(self, bbox, orig_height, orig_width, factor=28, min_pixels=28*28*256, max_pixels=14*14*4*1280):
        new_height, new_width = self.smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
        scale_w = new_width / orig_width
        scale_h = new_height / orig_height
        
        x1, y1, x2, y2 = bbox
        x1_new = round(x1 * scale_w)
        y1_new = round(y1 * scale_h)
        x2_new = round(x2 * scale_w)
        y2_new = round(y2 * scale_h)
        
        x1_new = max(0, min(x1_new, new_width - 1))
        y1_new = max(0, min(y1_new, new_height - 1))
        x2_new = max(0, min(x2_new, new_width - 1))
        y2_new = max(0, min(y2_new, new_height - 1))
    
        return [x1_new, y1_new, x2_new, y2_new]

    def convert_point_to_qwen25vl_format(self, point, orig_height, orig_width, factor=28, min_pixels=28*28*256, max_pixels=14*14*4*1280):
        # Step 1: 計算 resize 後的尺寸
        new_height, new_width = self.smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
        
        scale_w = new_width / orig_width
        scale_h = new_height / orig_height
        x, y = point
        x_new = round(x * scale_w)
        y_new = round(y * scale_h)

        x_new = max(0, min(x_new, new_width - 1))
        y_new = max(0, min(y_new, new_height - 1))

        return [x_new, y_new]
    
    def get_sample(self, idx):
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_list = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_list = None
        item['img_url_abs'] = image_path

        # Set a fixed random seed
        random.seed(idx)
        element_idx = random.randint(0, len(item['element']) - 1)
        task = item['element'][element_idx]['instruction']
        # log_message(f"Eval: {idx} 圖片:對應到 {element_idx} instruction, {task}")

        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        source = screenspot_to_qwen(task, img_dict, self.xy_int, think_grounding=self.think_grounding)
        prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt",
                                        training=not self.inference)

        # pdb.set_trace()
        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"],
            image_sizes=data_dict_q["image_grid_thw"],
            labels=data_dict_q["labels"][0],
        )
        
        # set only keep selected element
        item['element'] = [item['element'][element_idx]]
        return (
            data_dict,
            item,
        )