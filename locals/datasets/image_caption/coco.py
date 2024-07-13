from email.policy import default
from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from collections import defaultdict
from constants import *

class COCOI2TDataset(Dataset):

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
                 output_mode: str = 'text',
                 raw_image: bool = False,
                 sample_mode: str='text',
                 shuffle: bool=False,
                 inference: bool=False,
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.inference = inference
        self.meta = js.load(open(path))
        if isinstance(self.meta, dict):
            self.meta = self.meta['annotations']
        self.sample_mode = sample_mode
        if self.sample_mode == 'image':
            tmp = defaultdict(list)
            for item in self.meta:
                tmp[item['image_id']].append(item['caption'])
            self.meta = [{'image_id': k, 'captions': v} for k,v in tmp.items()]
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.shuffle = shuffle # shuffle is for interleaved image-text data

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
        print(f"COCO has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def get_image_fn(self, image_id):
        return os.path.join(self.image_folder, 'train2014', 'COCO_train2014_{:012d}.jpg'.format(image_id))
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        item = self.meta[i]
        tgt_img = Image.open(self.get_image_fn(item['image_id'])).convert('RGB')
        if self.sample_mode == 'text':
            instruction = item['caption']
        elif self.sample_mode == 'image':
            instruction = random.choice(item['captions'])
        else:
            raise ValueError
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