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

class LvisCapDataset(Dataset):
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
                 shuffle: bool=False,
                 shuffle_prob: float=0.5,
                 inference: bool=False,
                 caption_type: str='long',
                 task_type: str='i2t',
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.inference = inference
        self.meta = js.load(open(path))
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.shuffle = shuffle # shuffle is for interleaved image-text data
        self.shuffle_prob = shuffle_prob # the probability for shuffle

        self.flip_prob = flip_prob
        self.task_type = task_type
        self.sample_weight = sample_weight
        self.caption_type = caption_type
        self.i2t_long_prompts = load_prompt('locals/datasets/prompts/prompt_captioning_long.txt')
        self.i2t_short_prompts = load_prompt('locals/datasets/prompts/prompt_captioning_short.txt')
        self.t2i_prompts = load_prompt('locals/datasets/prompts/prompt_txt2img.txt')
        print(f"LVIS-GPT4V-Captions has {len(self)} samples!!")

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
        tgt_img = Image.open(self.get_image_fn(item['image'])).convert('RGB')

        # find the caption
        if self.caption_type == 'random':
            prob = random.random()
            if prob > 0.5:
                caption_type = 'long'
            else:
                caption_type = 'short'
        else:
            caption_type = self.caption_type
        if caption_type == 'long':
            caption = item['caption']
            prompt = random.choice(self.i2t_long_prompts)
        elif caption_type == 'short':
            caption = item['short_caption']
            prompt = random.choice(self.i2t_short_prompts)
        else:
            raise ValueError

        if self.output_mode == 'conversation':
            if self.shuffle:
                prob = random.random()
                if prob > self.shuffle_prob:
                    task_type = 'i2t'
                else:
                    task_type = 't2i'
            else:
                task_type = self.task_type
            if task_type == 'i2t':
                instruction = '{} {}'.format(ALL_IMG_TOKENS_STR, prompt)
                response = caption
                label_masks = [0]
            elif task_type == 't2i':
                prompt = random.choice(self.t2i_prompts)
                instruction = '{} {}'.format(prompt, caption)
                response = ALL_IMG_TOKENS_STR
                label_masks = [1]
            else:
                raise ValueError
            sources = [{'from': 'human', 'value': instruction},
                    {'from': 'gpt', 'value': response}]
            return {'input_imges': [tgt_img], 'conversation': sources, 'image_label_masks': label_masks}
        elif self.output_mode == 'text':
            # raise ValueError
            if self.shuffle:
                prob = random.random()
                if prob > self.shuffle_prob:
                    task_type = 'i2t'
                else:
                    task_type = 't2i'
            else:
                task_type = self.task_type
            if task_type == 'i2t':
                text = "{} {}".format(ALL_IMG_TOKENS_STR, caption)
                image_label_masks = [0]
            elif task_type == 't2i':
                text = "{} {}".format(caption, ALL_IMG_TOKENS_STR)
                image_label_masks = [1]
            else:
                raise ValueError
            if self.inference:
                text = ALL_IMG_TOKENS_STR
            return {'input_images': [tgt_img], 'text': text, 'image_label_masks': image_label_masks}