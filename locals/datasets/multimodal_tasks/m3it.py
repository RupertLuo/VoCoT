from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from typing import List
from constants import *
import re, copy
from base64 import b64encode, b64decode
import io
from datasets import load_dataset, concatenate_datasets

def byte2image(byte_data):
    """
    convert byte to PIL image
    """
    if isinstance(byte_data, str):
        byte_data = b64decode(byte_data)
    image = Image.open(io.BytesIO(byte_data))
    return image

class M3ITDataset(Dataset):

    def __init__(self, 
                 path: str,
                 dataset_names: List[str],
                 split: str = 'train',
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 shuffle: bool = False,
                 raw_image: bool = False,
                 inference: bool = False,
                 min_size: int = 50, 
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.inference = inference
        tmp_ds = []
        for name in dataset_names:
            print(name)
            tmp_ds.append(load_dataset(os.path.join(path, name))[split])
        self.meta = concatenate_datasets(tmp_ds)
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.shuffle = shuffle
        self.min_size = min_size

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        print(f"LLaVA Academic has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        item = self.meta[i]
        image = item['image_base64_str']
        image = byte2image(image[0]).convert('RGB')

        if len(item['inputs']):
            prompt = item['instruction'] + ' ' + item['inputs']
        else:
            prompt = item['instruction']
        
        prompt = ALL_IMG_TOKENS_STR + ' ' + prompt
        conversation = [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': item['outputs']}
        ]
        return {'input_images': [image], 'conversation': conversation, 'image_label_masks': [0]}

        