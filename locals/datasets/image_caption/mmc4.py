from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from constants import *
import time
import os
from pathlib import Path
class FilteredMMC4Dataset(Dataset):

    def __init__(self, 
                 path: str,
                 image_folder: str = '',
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 raw_image: bool = False,
                 meta_folder: str = None,
                 expand2square: bool = False,
                 avoid_image_gen: bool=False,
                 **kwargs):

        time_start = time.time()
        self.path = path
        self.instruct = instruct
        self.meta_folder = meta_folder
        self.expand2square = expand2square
        self.avoid_image_gen = avoid_image_gen

        if os.path.isdir(path):
            self.meta = list(Path(path).rglob('*.json'))
            self.need_reload = False
        elif os.path.isfile(path) and path.endswith('jsonl'):
            self.meta_file = open(path)
            self.meta = []
            _ = 0
            for line in self.meta_file:
                self.meta.append(js.loads(line))
                _ +=1
            self.need_reload = False
        elif os.path.isfile(path) and path.endswith('txt'):
            self.meta = []
            with open(path) as rf:
                for line in rf:
                    self.meta.append(line.strip())
            self.need_reload = True
            assert self.meta_folder is not None
        else:
            raise ValueError('Invalid data path')
        # self.meta_file = open(path)
        # self.meta = []
        # _ = 0
        # for line in self.meta_file:
        #     self.meta.append(js.loads(line))
        #     _ +=1

            #### <<<<< Delete follow code for real training >>>> ####
            
            # if _ > 5000:
            #     break

            #########################################################

        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        time_end = time.time()
        print(f"MMC4 Dataset has {len(self)} samples!!, initialized with {time_end-time_start}")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        item = self.meta[i]
        # if item is a path
        if isinstance(item,Path):
            item = js.load(open(item))
        elif self.need_reload:
            item = js.load(open(os.path.join(self.meta_folder, item)))

        tgt_img_list = []
        img_txt_match_index = []
        for img_info in item['image_info']:
            tgt_img_list.append(Image.open(os.path.join(self.image_folder,img_info['img_path'])).convert('RGB'))
            img_txt_match_index.append(img_info['matched_text_index'])
        text_list = [txt for txt in item['text_list']]
        img_label_masks = []
        for i,match_index in enumerate(img_txt_match_index):
            thr = random.random()
            if i == 0 and thr <= 0.5:
                img_label_masks.append(0)
            else:
                if self.avoid_image_gen:
                    img_label_masks.append(0)
                else:
                    img_label_masks.append(1)
            text_list[match_index] = "{} {}".format(text_list[match_index], ALL_IMG_TOKENS_STR) if thr > 0.5 else "{} {}".format(ALL_IMG_TOKENS_STR, text_list[match_index])
        # return image_0, image_1, instruction

        assert self.output_mode == 'text'
        return {'input_images': tgt_img_list, 'text': ' '.join(text_list) , 'image_label_masks': img_label_masks}


if __name__ == "__main__":
    dataset = FilteredMMC4Dataset('/mnt/bn/luoruipu-disk/meta_data/mmc4/filter_mmc4_meta_with_img_abs_path.jsonl')
    print(repr(dataset[1000]))