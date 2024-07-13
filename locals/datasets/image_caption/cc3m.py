from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from constants import *

class FilteredCC3MI2TDataset(Dataset):

    def __init__(self, 
                 path: str,
                 image_folder: str,
                 meta_folder: str=None,
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 shuffle: bool = False,
                 raw_image: bool = False,
                 inference: bool = False, 
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.inference = inference
        if path.endswith('txt'):
            self.meta = []
            with open(path) as rf:
                for line in rf:
                    self.meta.append(line.strip())
            self.need_reload = True
        else:
            self.need_reload = False
            self.meta = js.load(open(path))
        if meta_folder is not None:
            self.meta_folder = os.path.join(os.path.dirname(path), 'meta')
        else:
            self.meta_folder = meta_folder
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.shuffle = shuffle

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        self.generation_prompts = [
                "generate image with caption:",
                "can you give me the image with caption:",
                "help me to generate this image:",
                "generate image with according to caption:",
                "according to caption, generate image:",
                "an image with caption:",
                "can you visualize this caption:",
            ]
        print(f"CC3MDataset has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        if self.need_reload:
            meta_fn = self.meta[i]
            item = js.load(open(os.path.join(self.meta_folder, meta_fn)))
        else:
            item = self.meta[i]
        # item = self.meta[i]
        tgt_img = Image.open(os.path.join(self.image_folder,item['image'])).convert('RGB')
        instruction = item['conversations'][1]['value']
        # return image_0, image_1, instruction

        if self.output_mode == 'conversation':
            sources = [{'from': 'human', 'value': random.choice(self.generation_prompts)+instruction},
                    {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
            return {'output_images': [tgt_img], 'conversation': sources, 'image_label_masks': [1]}
        elif self.output_mode == 'text':
            if self.shuffle:
                prob = random.random()
                if prob > 0.5:
                    text = "{} {}".format(ALL_IMG_TOKENS_STR, instruction)
                    image_label_masks = [0]
                else:
                    text = "{} {}".format(instruction, ALL_IMG_TOKENS_STR)
                    image_label_masks = [1]
            else:
                text = "{} {}".format(ALL_IMG_TOKENS_STR, instruction)
                image_label_masks = [0]
            if self.inference:
                text = ALL_IMG_TOKENS_STR
            return {'input_images': [tgt_img], 'text': text, 'image_label_masks': image_label_masks}