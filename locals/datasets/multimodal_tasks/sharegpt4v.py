import os
from PIL import Image
from locals.datasets.multimodal_tasks.single_image_base import SingleImageDataset
from copy import deepcopy
from constants import ALL_IMG_TOKENS_STR

class Sharegpt4vDataset(SingleImageDataset):
    def __init__(self,
                path: str,
                image_folder: str,
                llava_pretrain_folder: str,
                coco_img_folder: str,
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
        self.llava_pretrain_folder = llava_pretrain_folder
        self.coco_img_folder = coco_img_folder
        print(f"Sharegpt4vDataset has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        
        i = self.get_sampler_index(i)
        
        item = self.meta[i]
        if 'image' in item:
            image_fn = item['image']
            sub_dir = image_fn.split('/')[0]
            
            if sub_dir == 'llava':
                image_folder = self.llava_pretrain_folder
                image_fn = '/'.join(image_fn.split('/')[3:])
            elif sub_dir == 'coco':
                image_folder = self.coco_img_folder
                image_fn = '/'.join(image_fn.split('/')[1:])
            else:
                image_folder = self.image_folder
            tgt_img = [Image.open(os.path.join(image_folder, image_fn)).convert('RGB')] 

            new_conversation = deepcopy(item['conversations'])
            for conv in new_conversation:
                if '<image>' in conv['value']:
                    conv['value'] = conv['value'].replace('<image>', ALL_IMG_TOKENS_STR)

            return {'input_images': tgt_img, 'conversation': new_conversation,'id':'Sharegpt4v_'+item['id'],'image_path':item['image'],'image_label_masks': [0]}
        else:
            tgt_img = None
            return {'conversation': item['conversations'],'id':'Sharegpt4v_'+item['id'],'image_path':item['image']}

if __name__ == '__main__':
    from tqdm import tqdm
    test_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json'
    image_folder = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/ShareGPT4V/data/'
    coco_image_folder = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/COCO2017'
    llava_pretrain_folder = '/mnt/bn/luoruipu-disk/meta_data/pretrain_data/LLaVA-Pretrain/'
    test = Sharegpt4vDataset(path=test_path, image_folder=image_folder, llava_pretrain_folder=llava_pretrain_folder,coco_img_folder=coco_image_folder, output_mode = 'conversation')
    print(len(test))
    for i in tqdm(range(len(test))):
        test[i]
        