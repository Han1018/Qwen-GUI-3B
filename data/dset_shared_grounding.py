import os
import cv2
import pdb
import json
import random
import numpy as np
from PIL import Image
import math

import torch
import torch.nn.functional as F

import sys
sys.path.append('.')
from data.template import grounding_to_qwen, batch_add_answer, batch_add_answer_append

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
        
class GroundingDataset(torch.utils.data.Dataset):
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

        self.base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get('samples_per_epoch', 1)
        self.sample_prob = np.array([args_dict.get('text2point', 1), 
                                        args_dict.get('text2bbox', 0), 
                                        args_dict.get('point2text', 0), 
                                        args_dict.get('bbox2text', 0)])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()
        self.random_sample = args_dict.get('random_sample', False)
        
        self.num_turn = args_dict.get('num_turn', 1)
        self.shuffle_image_token = args_dict.get('shuffle_image_token', False)
        self.uniform_prompt = args_dict.get('uniform_prompt', False)
        self.crop_min = args_dict.get('crop_min', 1)
        self.crop_max = args_dict.get('crop_max', 1)
        self.xy_int = args_dict.get('xy_int', False)
        self.model_id = args_dict.get('model_id', 'Qwen/Qwen2-VL-2B-Instruct')
        self.think_grounding = args_dict.get('think_grounding', False)
        
        self.dataset_name = dataset

        print(f"Dataset: {dataset}; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        if self.random_sample:
            return len(self.json_data)
            # return self.samples_per_epoch
        else:
            return len(self.json_data)

    def __getitem__(self, idx):
        while True:
            try:
                return self.get_sample(idx)
            except Exception as e:
                # this is acceptable during training
                print(e)
                idx = random.randint(0, len(self.json_data) - 1)

    def random_crop_metadata(self, img, metadata, scale_range=(0.5, 1.0)):
        original_width, original_height = metadata['img_size']
        img_copy = img.copy()
        
        scale_w = random.uniform(*scale_range)
        scale_h = random.uniform(*scale_range)

        crop_width = int(original_width * scale_w)
        crop_height = int(original_height * scale_h)

        pad_x = pad_y = 0

        if crop_width > original_width or crop_height > original_height:
            pad_x = max(0, (crop_width - original_width) // 2)
            pad_y = max(0, (crop_height - original_height) // 2)

            padded_img = Image.new('RGB', (crop_width, crop_height), (255, 255, 255))
            padded_img.paste(img_copy, (pad_x, pad_y))

            img = padded_img
            img_width, img_height = crop_width, crop_height
        else:
            img_width, img_height = original_width, original_height

        crop_x_min = random.randint(0, img_width - crop_width)
        crop_y_min = random.randint(0, img_height - crop_height)
        crop_x_max = crop_x_min + crop_width
        crop_y_max = crop_y_min + crop_height

        cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        new_elements = []
        for element in metadata['element']:
            bbox = element['bbox']
            point = element['point']

            bbox_abs = [int(bbox[0] * original_width) + pad_x, int(bbox[1] * original_height) + pad_y,  
                        int(bbox[2] * original_width) + pad_x, int(bbox[3] * original_height) + pad_y]
            point_abs = [int(point[0] * original_width) + pad_x, int(point[1] * original_height) + pad_y]

            if (bbox_abs[0] >= crop_x_min and bbox_abs[2] <= crop_x_max and
                bbox_abs[1] >= crop_y_min and bbox_abs[3] <= crop_y_max):
                
                new_bbox = [(bbox_abs[0] - crop_x_min) / crop_width,
                            (bbox_abs[1] - crop_y_min) / crop_height,
                            (bbox_abs[2] - crop_x_min) / crop_width,
                            (bbox_abs[3] - crop_y_min) / crop_height]
                new_point = [(point_abs[0] - crop_x_min) / crop_width,
                            (point_abs[1] - crop_y_min) / crop_height]

                new_element = element.copy()
                new_element['bbox'] = new_bbox
                new_element['point'] = new_point
                new_elements.append(new_element)

        if len(new_elements) == 0:
            return img_copy, metadata

        metadata['element'] = new_elements
        metadata['element_size'] = len(new_elements)
        metadata['img_size'] = cropped_img.size
        return cropped_img, metadata
    
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
        # if not self.inference and self.random_sample:
        #     idx = random.randint(0, len(self.json_data) - 1)
        assert idx == idx % len(self.json_data)
        idx = idx % len(self.json_data)
        
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_list = [Image.open(image_path).convert("RGB")]
            if self.crop_min != 1 or self.crop_max != 1:
                image_list[0], item = self.random_crop_metadata(image_list[0], item, scale_range=(self.crop_min, self.crop_max))
        else:
            image_path = ""
            image_list = None

        # text2point, point2text, text2bbox, bbox2text
        sample_io = np.random.choice(len(self.sample_prob), p=self.sample_prob)

        # prepare for multi-turn streaming
        element_list = item['element']
        k = min(self.num_turn, len(element_list))
        assert k > 0
        # random.seed(idx) # for reproducibility
        element_cand = random.choices(element_list, k=k)
        if self.num_turn == 1:
            random.seed(idx)
            element_idx = random.randint(0, len(item['element']) - 1)
            element_cand = [item['element'][element_idx]]
            # log_message(f"training: {self.dataset_name}, {idx} 圖片:對應到 {element_idx} instruction, {element_cand[0]['instruction']}")

        if len(element_cand) == 1:
            element = element_cand[0]
            element_name = element['instruction']
            answer_xy = element['point'] if sample_io in [0, 2] else element['bbox']
            if self.xy_int:
                answer_xy = [int(x * 1000) for x in answer_xy]
            else:
                answer_xy = [round(x, 2) for x in answer_xy]
            if self.model_id == 'Qwen/Qwen2.5-VL-3B-Instruct':
                element['point'] = self.convert_point_to_qwen25vl_format(element['point'], item['img_size'][1], item['img_size'][0], min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                
            if self.think_grounding:
                answer_xy = f"""```json
{element['point']}
```
{element['description']}"""

            if sample_io in [2, 3]:
                element_name, answer_xy = answer_xy, element_name

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_qwen(element_name, img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt, self.think_grounding)
            prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
            data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt", training=not self.inference)
            data_dict_qa, answer = batch_add_answer(data_dict_q, answer_xy, self.processor)
            
        else:
            # multi-turn streaming
            element_name_list = [element['instruction'] for element in element_cand]
            if sample_io in [0, 2] and self.model_id == 'Qwen/Qwen2.5-VL-3B-Instruct':
                # transformed by convert_point_to_qwen25vl_format
                element_cand_qwen25 = [element.copy() for element in element_cand]
                for i, element in enumerate(element_cand_qwen25):
                    element['point'] = self.convert_point_to_qwen25vl_format(element['point'], item['img_size'][1], item['img_size'][0], min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                
                if self.think_grounding:
                    answer_xy_list = []
                    for element in element_cand_qwen25:
                        answer = f"""```json
{element['point']}
```
{element['description']}"""
                        answer_xy_list.append(answer)
                else:
                    answer_xy_list = [element['point'] for element in element_cand_qwen25]
            elif sample_io in [0, 2]:
                answer_xy_list = [element['point'] for element in element_cand]
            elif sample_io in [1, 3] and self.model_id == 'Qwen/Qwen2.5-VL-3B-Instruct':
                # transformed by convert_bbox_to_qwen25vl_format
                element_cand_qwen25 = [element.copy() for element in element_cand]
                for i, element in enumerate(element_cand_qwen25):
                    element['bbox'] = self.convert_bbox_to_qwen25vl_format(element['bbox'], item['img_size'][1], item['img_size'][0], min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                answer_xy_list = [element['bbox'] for element in element_cand_qwen25]
            else:
                answer_xy_list = [element['bbox'] for element in element_cand]

            if self.xy_int:
                answer_xy_list = [[int(x * 1000) for x in answer_xy] for answer_xy in answer_xy_list]
            else:
                if not self.think_grounding:
                    answer_xy_list = [[round(x, 2) for x in answer_xy] for answer_xy in answer_xy_list]

            if sample_io in [2, 3]:
                element_name_list, answer_xy_list = answer_xy_list, element_name_list
            answer_xy = answer_xy_list[0]

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_qwen(element_name_list[0], img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt, self.think_grounding)
            prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True) 
            data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt", training=not self.inference)
            data_dict_qa, answer = batch_add_answer_append(data_dict_q, answer_xy, self.processor,
                                                            append_element=element_name_list[1:], 
                                                            append_answer=answer_xy_list[1:])

        max_seq_len = self.processor.tokenizer.model_max_length
        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
            image_sizes=data_dict_qa["image_grid_thw"],
            pixel_values=data_dict_qa["pixel_values"],
            labels=data_dict_qa["labels"][0],        
        )
        assert data_dict_qa["input_ids"][0].shape[0] <= max_seq_len, f"Input seq. surpasses Max. seq len: {data_dict_qa['input_ids'][0].shape[0]} > {max_seq_len}"

        # Prepare elements for ShowUI
        for key in ['select_mask', 'patch_pos', 'patch_assign', 'patch_assign_len']:
            if key in data_dict_q:
                data_dict[key] = data_dict_q[key]

        return (
            data_dict,
            item,
        )

if __name__ == '__main__':
    from model.showui.processing_showui import ShowUIProcessor
    from model.showui.modeling_showui import ShowUIForConditionalGeneration

    processor = ShowUIProcessor.from_pretrained(
                                            "Qwen/Qwen2-VL-2B-Instruct", 
                                            min_pixels=1024*28*28, 
                                            max_pixels=1024*28*28,
                                            model_max_length=4096,
                                            uigraph_train=True, uigraph_test=True,
                                            uigraph_diff=1,  uigraph_rand=False,
                                            uimask_pre=True, uimask_ratio=1, uimask_rand=False
                                            )

    dataset = GroundingDataset(
        "/blob/v-lqinghong/data/GUI_database",
        "rico",
        "hf_train_rico",
        processor,
        inference=False
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])
        print(i, len(data[0]['input_ids']), data[0]['patch_assign_len'], data[0]['select_mask'].sum())
