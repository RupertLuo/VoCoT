from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from constants import ALL_IMG_TOKENS_STR
from locals.datasets.multimodal_tasks.single_image_base import SingleImageDataset
from copy import deepcopy

class LVISinstruct4vDataset(SingleImageDataset):
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
        super().__init__(path, image_folder, instruct, min_resize_res, max_resize_res, crop_res, flip_prob, sample_weight, check, output_mode, shuffle, raw_image, inference, min_size, **kwargs)
        print(f"LVIS-instruct4v has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        
        i = self.get_sampler_index(i)
        
        item = self.meta[i]
        if 'image' in item:
            image_fn = item['image']
            dirname = image_fn.split('/')[0]
            image_fn = '/'.join(image_fn.split('/')[1:])
            all_sub_images = []
            if dirname == 'coco':
                dirname = 'COCO2017'
            else:
                raise ValueError

            tgt_img = [Image.open(os.path.join(self.image_folder, dirname, image_fn)).convert('RGB')] 
            
            new_conversation = deepcopy(item['conversations'])
            for conv in new_conversation:
                if '<image>' in conv['value']:
                    conv['value'] = conv['value'].replace('<image>', ALL_IMG_TOKENS_STR)
            return {'input_images': tgt_img + all_sub_images, 'conversation': new_conversation,'id':'LVIS_'+item['id'],'image_label_masks': [0]}
        else:
            tgt_img = None
            return {'conversation': item['conversations'],'id':'LVIS_'+item['id']}

if __name__ == '__main__':
    test_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/LVIS-Instruct4V/lvis_instruct4v_220k.json'
    image_folder = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/'
    test = LVISinstruct4vDataset(path=test_path, image_folder=image_folder, output_mode = 'conversation')
    print(len(test))
    print(test[1000])


        