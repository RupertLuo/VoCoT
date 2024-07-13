from torch.utils.data import Dataset
import math
import random
from PIL import Image
# 使用 read_parquet 加载parquet文件
from pandas import read_parquet
import io

from constants import ALL_IMG_TOKENS_STR

class LlavaRInstructDataset(Dataset):
    def __init__(self,
                path: str,
                instruct: bool = False,
                min_resize_res: int = 256,
                max_resize_res: int = 256,
                crop_res: int = 256,
                flip_prob: float = 0.5,
                sample_weight: float = 1.0,
                check: bool = False,
                output_mode: str = 'conversation',
                shuffle: bool = False,
                raw_image: bool = False,
                inference: bool = False,
                min_size: int = 50, 
                **kwargs):
        self.path = path
        self.instruct = instruct
        self.inference = inference
        self.meta = read_parquet(path)
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
        print(f"LlavaR Instruct has {len(self)} samples!!")
    
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
        assert self.output_mode == 'conversation'
        ini_i = i
        i = self.get_sampler_index(i)
        item = self.meta.iloc[i]
        image_bytes = item['image']['bytes']
        tgt_img = [Image.open(io.BytesIO(image_bytes))]
        new_conversation = []
        assert len(item['user_texts']) == len(item['bot_texts'])
        for i in range(len(item['user_texts'])):
            if i == 0:
                new_conversation.append({'from':"human",'value': ALL_IMG_TOKENS_STR + item['user_texts'][i]})
            else:
                new_conversation.append({'from':"human",'value':item['user_texts'][i]})
            new_conversation.append({'from':"gpt",'value':item['bot_texts'][i]})
        
        return {'input_images': tgt_img, 'conversation': new_conversation,'id':f'llava_r_{ini_i}','image_label_masks': [0]}

if __name__ == '__main__':
    test_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/LLaVAR-Instruct-16K/data/train-00000-of-00001-890199abde0ec4ff.parquet'
    test = LlavaRInstructDataset(path=test_path)
    print(len(test))
    print(test[1000])
        
        