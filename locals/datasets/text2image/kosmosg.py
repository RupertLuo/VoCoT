from email.policy import default
from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from collections import defaultdict
from constants import *

def load_prompt(fn):
    prompts = []
    with open(fn) as f:
        line=f.readline()
        while line:
            line=line.strip('\n')
            prompts.append(line)
            line=f.readline()
    return prompts

class KosMosGDataset(Dataset):
    def __init__(self, 
                 path: str,
                 image_folder: str,
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'conversation',
                 raw_image: bool = False,
                 object_type: str = 'mask',
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.meta = js.load(open(path))
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        self.object_type = object_type
        self.prompts = load_prompt('locals/datasets/prompts/prompt_kosmosg.txt')
        print(f"KosMos-G has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def get_image_fn(self, image_id):
        return os.path.join(self.image_folder, image_id)
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        item = self.meta[i]
        caption = item['source'].replace('<image>', ' '+ALL_IMG_TOKENS_STR)
        tgt_img = Image.open(item['target']).convert('RGB')
        prompt = random.choice(self.prompts)

        # find the caption
        if self.object_type == 'random':
            prob = random.random()
            if prob > 0.5:
                object_type = 'mask'
            else:
                object_type = 'obj'
        else:
            object_type = self.object_type
        if object_type == 'mask':
            object_images = [Image.open(img).convert('RGB') for img in item['masks']]
        elif object_type == 'obj':
            object_images = [Image.open(img).convert('RGB') for img in item['objects']]
        else:
            raise ValueError

        if self.output_mode == 'conversation':
            sources = [{'from': 'human', 'value': prompt.format(desc=caption)},
                    {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
            return {'input_images': object_images+[tgt_img], 'conversation': sources, 'image_label_masks': [0]*len(object_images)+[1]}
        elif self.output_mode == 'text':
            text = caption + ' ' + '{}.'.format(ALL_IMG_TOKENS_STR)
            return {'input_images': object_images+[tgt_img], 'text': text, 'image_label_masks': [0]*len(object_images)+[1]}