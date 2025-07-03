import numpy as np
import torch

import pdb
import sys
sys.path.append(".")
from data.data_utils import IGNORE_INDEX
from data.dset_screenspot import ScreenSpotDataset
from data.dset_shared_grounding import GroundingDataset
from data.dset_trainingdata import TrainingDataset
import torch.distributed as dist

from transformers import AutoProcessor
import random  # 確保可以使用固定隨機種子

import pdb
def collate_fn(batch, processor=None):
    batch_data_list = [x[0] for x in batch]
    input_ids, labels = tuple([instance[key] for instance in batch_data_list]
                              for key in ("input_ids", "labels"))
    
    padding_value = processor.tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=padding_value)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :processor.tokenizer.model_max_length]
    labels = labels[:, :processor.tokenizer.model_max_length]

    # text-only
    if batch_data_list[0]['pixel_values'] is None:
        pixel_values = None
        image_sizes = None
    # vision
    elif len(batch_data_list[0]['pixel_values'].shape) == 2:
        pixel_values = [instance["pixel_values"] for instance in batch_data_list]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in batch_data_list]
        image_sizes = torch.cat(image_sizes, dim=0)

    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
    meta_data = [x[-1] for x in batch]
    result = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "image_sizes": image_sizes,
        "meta_data": meta_data,
    }

    for key in ['patch_assign', 'patch_assign_len']:
        if key in batch_data_list[0]:
            key_tmp = [instance[key] for instance in batch_data_list]
            key_tmp = torch.cat(key_tmp, dim=0)
            result[key] = key_tmp 
    for key, pad_val in zip(['patch_pos', 'select_mask'], [-1, True]):
        if key in batch_data_list[0]:
            key_tmp = [instance[key] for instance in batch_data_list]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=pad_val) for pos in key_tmp]
            key_tmp = torch.cat(padded_key, dim=0)
            result[key] = key_tmp
    return result

def log_message(message):
    log_file_path = "log.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{message}\n")
        
class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processor,
        inference,
        args,
        eval_train=False):

        self.inference = inference
        if self.inference:
            args.shuffle_image_token = False
            dset_list = args.val_dataset
            json_list = args.val_json
            sample_rate = args.val_ratio
        else:
            dset_list = args.train_dataset
            json_list = args.train_json
            sample_rate = args.train_ratio

        self.dset_list = [item for item in dset_list.split(",") if item]
        self.json_list = [item for item in json_list.split(",") if item]
        sample_rate = np.array([float(x) for x in sample_rate.split(",")])
        self.sample_rate = sample_rate / sample_rate.sum()
        # pdb.set_trace()
        
        if eval_train:
            self.dset_list = [args.eval_train_dataset]
            self.json_list = [args.eval_train_json]
            
        self.random_seed = 42
        self.sample_size = 500 # 5% training dataset size
        
        self.samples_per_epoch = args.samples_per_epoch
        self.random_sample = args.random_sample
        self.record_sample = args.record_sample
        self.uniform_sample = args.uniform_sample

        # pdb.set_trace()
        if self.inference:
            assert len(self.dset_list) == len(self.json_list) == 1
        else:
            assert len(self.dset_list) == len(self.json_list) == len(self.sample_rate)

        self.all_datasets = []
        for dataset, json_split in zip(self.dset_list, self.json_list):
            # Grounding SFT
            if dataset in ["showui-web","showui-desktop","amex","icon","icon_apple","icon_uitars", "web_omni", "explanation", "uground"] and self.inference == False:
                self.all_datasets.append(
                    GroundingDataset(
                        dataset_dir=args.dataset_dir,
                        dataset=dataset,
                        json_data=json_split,
                        processor=processor,
                        inference=False,
                        args_dict=vars(args),
                        )
                )
            # Zero-shot evaluation
            elif dataset == "screenspot" or dataset == "screenspot2":
                self.all_datasets.append(
                    ScreenSpotDataset(
                        dataset_dir=args.dataset_dir,
                        dataset=dataset,
                        json_data=json_split,
                        processor=processor,
                        inference=True,
                        args_dict=vars(args),
                        )
                )
            elif dataset != "screenspot" and self.inference == True:
                self.all_datasets.append(
                    TrainingDataset(
                        dataset_dir=args.dataset_dir,
                        dataset=dataset,
                        json_data=json_split,
                        processor=processor,
                        inference=True,
                        args_dict=vars(args),
                        )
                )
        
        # 固定隨機種子抽樣
        if self.inference:
            for i, dataset in enumerate(self.all_datasets):
                if len(dataset) > 2000:
                    random.seed(self.random_seed)
                    sampled_indices = random.sample(range(len(dataset)), self.sample_size)
                    self.all_datasets[i] = torch.utils.data.Subset(dataset, sampled_indices)

        self.sample_recorder = [set() for _ in range(len(self.all_datasets))]
        if self.inference:
            print(f"Loading {len(self.dset_list)} Validation Datasets")
        else:
            print(f"Loading {len(self.dset_list)} Training Datasets")

    def __len__(self):
        total_len = sum([len(dataset) for dataset in self.all_datasets])
        if self.inference:
            return total_len

        if self.random_sample:
            return self.samples_per_epoch
        else:
            return total_len

    def refresh_sample_recorder(self, ind):
        print(f"Refresh sample recorder for dataset {ind}")
        self.sample_recorder[ind] = set()
        
    def log_message(message):
        log_file_path = "log_datasets.txt"
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{message}\n")

    def __getitem__(self, idx):
        # A: Prob. sample + fixed steps                 -- random_sample
        # B: balanced sampling;                         -- random_sample & record_sample
        # C: Concat sample unrepeated once;             -- not random_sample
        # D:                        -- new logic
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # log_message(f"[GPU {rank}] getting index = {idx}")

        if not self.inference:
            if self.uniform_sample:
                # pdb.set_trace()
                sample_rate = np.array([len(x) for x in self.all_datasets])
                self.sample_rate = sample_rate / sample_rate.sum()
                print(f"Using uniform sampling with default ratio: {self.sample_rate}.")

            # training with randomly sampling
            if self.random_sample and not self.record_sample:
                ind = np.random.choice(list(range(len(self.dset_list))), p=self.sample_rate)
                data = self.all_datasets[ind]

                data_main, data_meta = data[idx]
                return data_main, data_meta
            
            # training with round-robin sampling
            elif self.random_sample and self.record_sample:
                # Initialize a pointer to track the current dataset index if not already done
                if not hasattr(self, 'current_dataset_index'):
                    self.current_dataset_index = 0

                # Get the current dataset index
                ind = self.current_dataset_index
                data = self.all_datasets[ind]

                # Get unseen indices for the current dataset
                unseen_indices = set(range(len(data))) - self.sample_recorder[ind]
                if not unseen_indices:
                    log_message(f"training: {data.dataset_name}, reset")
                    self.refresh_sample_recorder(ind)
                    unseen_indices = set(range(len(data)))

                # Randomly select a sample from the unseen indices
                sample_idx = np.random.choice(list(unseen_indices))
                self.sample_recorder[ind].add(sample_idx)
                
                # Move to the next dataset in a round-robin fashion
                self.current_dataset_index = (self.current_dataset_index + 1) % len(self.dset_list)

                # Return the selected sample
                data_main, data_meta = data[sample_idx]
                return data_main, data_meta

            # training with all datasets concated once
            elif not self.random_sample:
                if idx >= len(self):
                    raise IndexError
                for data in self.all_datasets:
                    if idx < len(data):
                        data_main, data_meta = data[idx]
                        return data_main, data_meta
                    else:
                        idx -= len(data)

        # evaluation with only one dataset;
        elif self.inference:
            assert len(self.all_datasets) == 1
            data = self.all_datasets[0]
            data_main, data_meta = data[idx]
            return data_main, data_meta
