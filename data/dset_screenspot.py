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
    "screenspot": "ScreenSpot",
    "screenspot2": "ScreenSpot-v2",
}

class ScreenSpotDataset(torch.utils.data.Dataset):
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

        # self.base_image_dir = os.path.join(dataset_dir, 'ScreenSpot')
        self.base_image_dir = os.path.join(dataset_dir, dataset_mapping[dataset])
        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get('samples_per_epoch', 1)
        self.xy_int = args_dict.get('xy_int', False)
        text2bbox = args_dict.get('text2bbox', 0)
        self.isbbox = (text2bbox == 1)
        self.think_grounding = args_dict.get('think_grounding', False)

        print(f"Dataset: Screenspot; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def get_sample(self, idx):
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_list = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_list = None
        item['img_url_abs'] = image_path

        task = item['task']
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        source = screenspot_to_qwen(task, img_dict, self.xy_int, self.isbbox, self.think_grounding)
        prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt",
                                        training=not self.inference)

        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"],
            image_sizes=data_dict_q["image_grid_thw"],
            labels=data_dict_q["labels"][0],
        )

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
                                            uigraph_train=True, uigraph_test=False,
                                            uigraph_diff=1,  uigraph_rand=False,
                                            uimask_pre=True, uimask_ratio=1, uimask_rand=False
                                            )

    dataset = ScreenSpotDataset(
        "/blob/v-lqinghong/data/GUI_database",
        "ScreenSpot",
        "hf_test_full",
        processor,
        inference=True
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])
        print(i, len(data[0]['input_ids']), data[0]['patch_assign_len'], data[0]['select_mask'].sum())