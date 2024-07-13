from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image

class SingleImageDataset(Dataset):

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
                 shuffle: bool = False,
                 raw_image: bool = False,
                 inference: bool = False,
                 min_size: int = 50, 
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
        self.shuffle = shuffle
        self.min_size = min_size
        self.flip_prob = flip_prob
        self.sample_weight = sample_weight

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def get_sampler_index(self,i):
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        return i
    
    def __getitem__(self, i):
        raise NotImplementedError