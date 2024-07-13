import os
from PIL import Image
from locals.datasets.multimodal_tasks.single_image_base import SingleImageDataset
from constants import ALL_IMG_TOKENS_STR
import random
class MidjourneyDataset(SingleImageDataset):
    def __init__(self,
                path: str,
                image_folder: str,
                instruct_prompt_path: str = None,
                instruct: bool = False,
                min_resize_res: int = 256,
                max_resize_res: int = 256,
                crop_res: int = 256,
                flip_prob: float = 0.5,
                sample_weight: float = 1.0,
                check: bool = False,
                shuffle: bool = False,
                raw_image: bool = False,
                inference: bool = False,
                min_size: int = 50, 
                **kwargs):
        output_mode = 'conversation'
        super().__init__(path, image_folder, instruct, min_resize_res, max_resize_res, crop_res, flip_prob, sample_weight, check, output_mode, shuffle, raw_image, inference, min_size, **kwargs)
        self.prompt_list = open(instruct_prompt_path).readlines()
        print(f"MidjourneyDataset has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        i = self.get_sampler_index(i)
        
        item = self.meta[i]

        image_fn = item['id']

        tgt_img = [Image.open(os.path.join(self.image_folder, image_fn+'.png')).convert('RGB')] 
        instruct = random.choice(self.prompt_list)
        new_conversation = [{'from': 'human', 'value': instruct + ','.join(item['event']['textPrompt'])},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]

        return {'output_images': tgt_img, 'conversations': new_conversation,'id':'midjourney_'+item['id'],'image_label_masks': [1]}
       

if __name__ == '__main__':
    from tqdm import tqdm
    test_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/midjourney/midjourney_sample_50k_new.json'
    image_folder = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/midjourney/images'
    instruct_prompt_path = '/mnt/bn/yangmin-priv/luoruipu/code/Edit-GPT4/locals/datasets/prompts/prompt_txt2img.txt'
    test = MidjourneyDataset(path=test_path, image_folder=image_folder,instruct_prompt_path=instruct_prompt_path)
    print(test[0])
    # import json as js
    # data = js.load(open(test_path))
    # new_data = []
    # for item in tqdm(data):
    #     image_fn = item['id']
    #     try:
    #         img = Image.open(os.path.join(image_folder, image_fn+'.png'))
    #         new_data.append(item)
    #     except:
    #         continue
    # print(len(new_data))
    # js.dump(new_data, open('/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/midjourney/midjourney_filtered.json','w'))