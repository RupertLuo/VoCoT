# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Tiankai Hang (tkhang@seu.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
import PIL
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import random

from locals.datasets.utils.zip_manager import MultipleZipManager
# from datasets import load_dataset
from constants import ALL_IMG_TOKENS_STR
import json as js
if hasattr(Image, "Resampling"):
    # deprecated in pillow >= 10.0.0
    RESAMPLING_METHOD = Image.Resampling.LANCZOS
else:
    RESAMPLING_METHOD = Image.LANCZOS


class FilteredIP2PDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        instruct: bool = False,
        max_num_images = None,
        sample_weight: float = 1.0,
        reverse_version: bool = False,
        raw_image: bool = True,
        **kwargs
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct
        self.raw_image = raw_image
        zip_list = []
        for i in range(zip_start_index, zip_end_index):
            name = "shard-"+str(i).zfill(2)+'.zip'
            zip_list.append(os.path.join(self.path, name))

        self.image_dataset = MultipleZipManager(zip_list, 'image', sync=True)   # sync=True is faster

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

        if max_num_images is not None and max_num_images > 0:
            self.seeds = self.seeds[:min(max_num_images, len(self.seeds))]

        # flatten seeds
        self.seeds = [(name, seed) for name, seeds in self.seeds for seed in seeds]
        self.sample_weight = sample_weight
        
        while True:
            try:
                with open('filtered_ids_ip2p.json') as json_file:
                    filtered_ids = json.load(json_file)
                break
            except:
                # download json file from url
                if reverse_version:
                    os.system('wget https://github.com/TiankaiHang/storage/releases/download/readout/filtered_ids_ip2p.json')
                else:
                    os.system("wget https://github.com/TiankaiHang/storage/releases/download/readout/filtered-ip2p-thres5.5-0.5.json -O filtered_ids_ip2p.json")
        
        print("seeds:", len(self.seeds))
        # self.seeds = [seed for seed in self.seeds if seed[1] in filtered_ids]
        # faster
        # self.seeds = list(filter(lambda seed: seed[1] in filtered_ids, self.seeds))
        # to numpy and faster in parallel
        # import pdb; pdb.set_trace()
        _seeds = [f"{a}/{b}" for a, b in self.seeds]
        self.seeds = np.array(self.seeds)
        _seeds = np.array(_seeds)
        self.seeds = self.seeds[np.isin(_seeds, filtered_ids)]
        self.seeds = self.seeds.tolist()

        self.return_add_kwargs = kwargs.get("return_add_kwargs", False)
        print(f"InstructP2P has {len(self)} samples!!")

    def __len__(self) -> int:
        return int(len(self.seeds) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:

        if self.raw_image:
            return self.get_raw_image(i)
        else:
            return self.get_processed_image(i)
    
    def get_processed_image(self, i):
        
        # name, seeds = self.seeds[i]
        if self.sample_weight >= 1:
            i = i % len(self.seeds)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        name, seed = self.seeds[i]
        propt_name = name + "/prompt.json"
        if not self.image_dataset.managers[self.image_dataset.mapping[propt_name]]._init:
            self.image_dataset.managers[self.image_dataset.mapping[propt_name]].initialize(close=False)
        # propt_name = name + "/prompt.json"
        byteflow = self.image_dataset.managers[self.image_dataset.mapping[propt_name]].zip_fd.read(propt_name)
        texts = json.loads(byteflow.decode('utf-8'))
        prompt = texts["edit"]
        if self.instruct:
            prompt = "Image Editing: " + prompt

        text_input = texts["input"]
        text_output = texts["output"]

        # image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        # image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        image_0 = self.image_dataset.get(name+f"/{seed}_0.jpg")
        image_1 = self.image_dataset.get(name+f"/{seed}_1.jpg")

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        if self.return_add_kwargs:
            add_kwargs = dict(
                name=name,
                seed=seed,
                text_input=text_input,
                text_output=text_output,
            )
        else:
            add_kwargs = {}

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt), **add_kwargs)

    def get_info(self, i):
        name, seed = self.seeds[i]
        propt_name = name + "/prompt.json"
        if not self.image_dataset.managers[self.image_dataset.mapping[propt_name]]._init:
            self.image_dataset.managers[self.image_dataset.mapping[propt_name]].initialize(close=False)
        # propt_name = name + "/prompt.json"
        byteflow = self.image_dataset.managers[self.image_dataset.mapping[propt_name]].zip_fd.read(propt_name)
        texts = json.loads(byteflow.decode('utf-8'))
        prompt = texts["edit"]

        # image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        # image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        image_0 = self.image_dataset.get(name+f"/{seed}_0.jpg")
        image_1 = self.image_dataset.get(name+f"/{seed}_1.jpg")
        return image_0, image_1, prompt
    
    def get_raw_image(self, i):
        
        # name, seeds = self.seeds[i]
        if self.sample_weight >= 1:
            i = i % len(self.seeds)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        name, seed = self.seeds[i]
        propt_name = name + "/prompt.json"
        if not self.image_dataset.managers[self.image_dataset.mapping[propt_name]]._init:
            self.image_dataset.managers[self.image_dataset.mapping[propt_name]].initialize(close=False)
        # propt_name = name + "/prompt.json"
        byteflow = self.image_dataset.managers[self.image_dataset.mapping[propt_name]].zip_fd.read(propt_name)
        texts = json.loads(byteflow.decode('utf-8'))
        prompt = texts["edit"]
        if self.instruct:
            prompt = "Image Editing: " + prompt

        text_input = texts["input"]
        text_output = texts["output"]

        # image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        # image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        image_0 = self.image_dataset.get(name+f"/{seed}_0.jpg")
        image_1 = self.image_dataset.get(name+f"/{seed}_1.jpg")

        prob = random.random()
        instruction = '{}: <Img><ImageHere></Img>.'.format(prompt) if prob < 0.5 else '<Img><ImageHere></Img> {}'.format(prompt)
        sources = [{'from': 'human', 'value': instruction},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
        return {'input_images': [image_0], 'output_images': [image_1], 'output_cond_images': [image_0], 'conversation': sources, 'image_label_masks': [0, 1]}

class ICP2PDataset(FilteredIP2PDataset):
    def __init__(
        self,
        meta_path: str,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        instruct: bool = False,
        max_num_images = None,
        sample_weight: float = 1.0,
        reverse_version: bool = False,
        raw_image: bool = True,
        **kwargs
    ):
        self.meta = json.load(open(meta_path))
        super(ICP2PDataset, self).__init__(path=path, split=split, splits=splits, min_resize_res=min_resize_res,
                                           max_resize_res=max_resize_res, crop_res=crop_res, flip_prob=flip_prob,
                                           zip_start_index=zip_start_index, zip_end_index=zip_end_index, instruct=instruct,
                                           max_num_images = max_num_images, sample_weight=sample_weight, reverse_version=reverse_version,
                                           raw_image=raw_image, **kwargs)
        print(f"In-Contexted InstructP2P has {len(self)} samples!!")

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        # name, seeds = self.seeds[i]
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        conflict_ids = self.meta[i]
        id_0, id_1 = np.random.choice(conflict_ids, 2, replace=False)
        image_0_0, image_0_1, prompt_0 = self.get_info(id_0)
        image_1_0, image_1_1, prompt_1 = self.get_info(id_1)

        assert prompt_0==prompt_1, "mismatched prompts: {}; {}".format(prompt_0, prompt_1)

        sources = [{'from': 'human', 'value': 'Given the transformation between <Img><ImageHere></Img> and <Img><ImageHere></Img>, apply the same editing to <Img><ImageHere></Img>.'},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
        return {'input_images': [image_0_0, image_0_1, image_1_0], 'output_images': [image_1_1], 'output_cond_images': [image_1_0], 'conversation': sources, 'image_label_masks': [0, 0, 0, 1]}



class GIERDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        request: str = 'both',
        check: bool = False,
        sample_weight: float = 1.0,
        instruct: bool = False,
        raw_image: bool = False
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct
        self.check = check
        self.raw_image = raw_image

        # self.meta = torch.load(Path(self.path, "GIER.json"), map_location="cpu")
        # load json file
        with open(Path(self.path, "GIER.json")) as json_file:
            self.meta = json.load(json_file)

        with open(Path(self.path, 'split.json'), 'r') as json_file:
            split_info = json.load(json_file)
            assert split in split_info, "{} split does not exist in GIER".format(split)
            self.meta = [self.meta[i] for i in split_info[split]]

        print(f"||||||||||||||||||||||||||||| \n Loaded {len(self.meta)} images from json file")

        # assert the instruction source
        assert request in ['both', 'expert', 'amateur'], 'the request must come from expert or amateur annotators'

        input_does_not_exist = []
        output_does_not_exist = []
        # filter out out images that do not exist
        if not os.path.exists(os.path.join(self.path, "filtered_meta_new.pt")):
            filtered_meta = []
            for i in tqdm(range(len(self.meta))):
                input_path = os.path.join(self.path, "warped", self.meta[i]["input"])
                output_path = os.path.join(self.path, "warped", self.meta[i]["output"])

                if not os.path.exists(input_path):
                    input_path = os.path.join(self.path, "images", self.meta[i]["input"])
                    if not os.path.exists(input_path):
                        input_does_not_exist.append(input_path)
                
                if not os.path.exists(output_path):
                    output_path = os.path.join(self.path, "images", self.meta[i]["output"])
                    if not os.path.exists(output_path):
                        output_does_not_exist.append(output_path)
                
                # check the instruction
                if request == 'both':
                    self.meta[i]["prompts"] = self.meta[i]["expert_summary"] + self.meta[i]["amateur_summary"]
                elif request == 'expert':
                    self.meta[i]["prompts"] = self.meta[i]["expert_summary"]
                elif request == 'amateur':
                    self.meta[i]["prompts"] = self.meta[i]["amateur_summary"]
                else:
                    raise ValueError
                
                if os.path.exists(input_path) and os.path.exists(output_path):
                    filtered_meta.append(
                        dict(
                            input=input_path,
                            output=output_path,
                            prompts=self.meta[i]["prompts"],
                        )
                    )
                else:
                    print(f"\n {input_path} or {output_path} does not exist")
            torch.save(filtered_meta, os.path.join(self.path, "filtered_meta_new.pt"))
        else:
            filtered_meta = torch.load(os.path.join(self.path, "filtered_meta_new.pt"), map_location="cpu")
        
        self.meta = filtered_meta
        print(f"||||||||||||||||||||||||||||| \n Filtered {len(self.meta)} images")
        for i in range(len(self.meta)):
            self.meta[i]['input'] = self.meta[i]['input'].replace('/mnt/external/datasets/GIER_editing_data/', self.path)
            self.meta[i]['output'] = self.meta[i]['output'].replace('/mnt/external/datasets/GIER_editing_data/', self.path)

        # write input_does_not_exist and output_does_not_exist to file
        # with open(Path(self.path, f"input_does_not_exist.txt"), "w") as f:
        #     for item in input_does_not_exist:
        #         f.write("%s\n" % item)
        # with open(Path(self.path, f"output_does_not_exist.txt"), "w") as f:
        #     for item in output_does_not_exist:
        #         f.write("%s\n" % item)
        
        # split_0, split_1 = {
        #     "train": (0.0, splits[0]),
        #     "val":   (splits[0], splits[0] + splits[1]),
        #     "test":  (splits[0] + splits[1], 1.0),
        # }[split]

        # idx_0 = math.floor(split_0 * len(self.meta))
        # idx_1 = math.floor(split_1 * len(self.meta))
        
        self.sample_weight = sample_weight
        print('original GIER', len(self.meta))
        print(f"Processed GIER has {len(self)} samples!!")

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, index):
        if self.raw_image:
            return self.get_raw_image(index)
        else:
            return self.get_processed_image(index)

    def get_processed_image(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        # prompt = self.meta[i]["prompts"]
        prompt = random.choice(self.meta[i]["prompts"])
        try:
            image_0 = Image.open(self.meta[i]["input"]).convert("RGB")
            image_1 = Image.open(self.meta[i]["output"]).convert("RGB")
        except PIL.UnidentifiedImageError:
            print(f"\n {self.meta[i]['input']} or {self.meta[i]['output']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        if self.instruct:
            prompt = "Image Editing: " + prompt

        if self.check:
            return dict(edited=Image.open(self.meta[i]["output"]).convert("RGB"), edit=dict(source=Image.open(self.meta[i]["input"]).convert("RGB"), instruction=prompt))
        return dict(edited=image_1, edit=dict(source=image_0, instruction=prompt))
    
    def get_raw_image(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        # prompt = self.meta[i]["prompts"]
        prompt = random.choice(self.meta[i]["prompts"])
        try:
            image_0 = Image.open(self.meta[i]["input"]).convert("RGB")
            image_1 = Image.open(self.meta[i]["output"]).convert("RGB")
        except PIL.UnidentifiedImageError:
            print(f"\n {self.meta[i]['input']} or {self.meta[i]['output']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)

        if self.instruct:
            prompt = "Image Editing: " + prompt

        prob = random.random()
        instruction = '{}: <Img><ImageHere></Img>.'.format(prompt) if prob < 0.5 else '<Img><ImageHere></Img> {}'.format(prompt)
        sources = [{'from': 'human', 'value': instruction},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
        return {'input_images': [image_0], 'output_images': [image_1], 'output_cond_images': [image_0], 'conversation': sources, 'image_label_masks': [0, 1]}


class FilteredCC3MDataset(Dataset):

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
                 raw_image: bool = False,
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

        item = self.meta[i]
        tgt_img = Image.open(os.path.join(self.image_folder,item['image']))
        instruction = item['conversations'][1]['value']
        # return image_0, image_1, instruction

        sources = [{'from': 'human', 'value': random.choice(self.generation_prompts)+instruction},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
        return {'output_images': [tgt_img], 'conversation': sources, 'image_label_masks': [1]}


class GQAInpaintDataset(Dataset):
    r"""
    shoud download and unzip the data first

    ```
    mkdir -p ../datasets
    cd ../datasets

    # if file exists, then skip
    if [ ! -f "gqa-inpaint.zip" ]; then
        sudo azcopy copy "https://bingdatawu2.blob.core.windows.net/genrecog/private/t-thang/gqa-inpaint.zip${TOKEN}" .
        unzip gqa-inpaint.zip -d gqa-inpaint > /dev/null
    fi

    if [ ! -f "images.zip" ]; then
        sudo azcopy copy "https://bingdatawu2.blob.core.windows.net/genrecog/private/t-thang/images.zip${TOKEN}" .
        unzip images.zip > /dev/null
    fi
    ```
    
    """
    def __init__(self, 
                 path: str,
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 raw_image: bool = False,
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        with open(os.path.join(self.path, "meta_info.json"), "r") as f:
            self.meta_info = json.load(f)

        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        print(f"GQA-Inpaint has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta_info) * self.sample_weight)
    
    def __getitem__(self, index):
        if self.raw_image:
            return self.get_raw_image(index)
        else:
            return self.get_processed_image(index)

    def get_processed_image(self, i):
        if self.sample_weight >= 1:
            i = i % len(self.meta_info)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        item = self.meta_info[i]
        edit_target = random.choice(item['edit'])
        src_img = Image.open(os.path.join(self.path, edit_target['source_image_path'])).convert("RGB")
        tgt_img = Image.open(os.path.join(self.path, item['target_image_path'])).convert("RGB")

        if edit_target['edit_type'] == 'remove':
            src_img, tgt_img = tgt_img, src_img

        image_0 = src_img
        image_1 = tgt_img

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)
        instruction = edit_target["prompt"]
        if self.instruct:
            instruction = "Image Editing: " + instruction
        # return image_0, image_1, instruction

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        if self.check:
            return dict(edited=tgt_img, edit=dict(source=src_img, instruction=instruction))
        return dict(edited=image_1, edit=dict(source=image_0, instruction=instruction))
    
    def get_raw_image(self, i):
        if self.sample_weight >= 1:
            i = i % len(self.meta_info)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        item = self.meta_info[i]
        edit_target = random.choice(item['edit'])
        src_img = Image.open(os.path.join(self.path, edit_target['source_image_path'])).convert("RGB")
        tgt_img = Image.open(os.path.join(self.path, item['target_image_path'])).convert("RGB")

        if edit_target['edit_type'] == 'remove':
            src_img, tgt_img = tgt_img, src_img

        image_0 = src_img
        image_1 = tgt_img

        instruction = edit_target["prompt"]
        if self.instruct:
            instruction = "Image Editing: " + instruction
        # return image_0, image_1, instruction

        sources = [{'from': 'human', 'value': '{}: <Img><ImageHere></Img>.'.format(instruction)},
                   {'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)}]
        return {'input_images': [image_0], 'output_images': [image_1], 'output_cond_images': [image_0], 'conversation': sources, 'image_label_masks': [0, 1]}



class MagicBrushDataset(Dataset):
    def __init__(
        self,
        path: str = "/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/MagicBrush/conversation_data_train.json",
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        len_dataset: int = -1,
        instruct: bool = False,
        sample_weight: float = 1.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct
        self.sample_weight = sample_weight

        self.meta = js.load(open(self.path, 'r'))
        print(f"MagicBrush has {len(self)} samples!!")
        

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        item = self.meta[i]
        try:
            image_0 = Image.open(item['source_img'],).convert("RGB") 
            image_1 = [Image.open(img).convert("RGB")  for img in item['target_img']]
        except (PIL.UnidentifiedImageError, FileNotFoundError):
            print(f"\n {self.path}/{item['input']} or {self.path}/{item['edited']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)
        prompt = item["instruction"]

        if self.instruct:
            prompt = ["Image Editing: " + pmt for pmt in prompt]
        # return image_0, image_1, prompt
        sources = []
        num_out_images = 0
        for i,pmt in enumerate(prompt):
            if i ==0:
                sources.append({'from': 'human', 'value': '{}: <Img><ImageHere></Img>.'.format(pmt)})
            else:
                sources.append({'from': 'human', 'value': pmt})

            sources.append({'from': 'gpt', 'value': '{}.'.format(ALL_IMG_TOKENS_STR)})
            num_out_images += 1

        return {'input_images': [image_0], 'output_images': image_1, 'output_cond_images': [image_0]+image_1[:-1], 'conversation': sources, 'image_label_masks': [0]+[1]*num_out_images}


class IEIWDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        zip_start_index: int = 0,
        zip_end_index: int = 30,
        sample_weight: float = 1.0,
        instruct: bool = False,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.instruct = instruct

        self.meta_path = os.path.join(self.path, "meta_infov1.json")
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        self.sample_weight = sample_weight
        print('original synthetic', len(self.meta))

    def __len__(self) -> int:
        return int(len(self.meta) * self.sample_weight)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1)

        item = self.meta[i]
        item['input'] = item['input'].replace('/mnt/external/tmp/2023/06/11/', self.path)
        item['edited'] = item['edited'].replace('/mnt/external/tmp/2023/06/11/', self.path)
        try:
            image_0 = Image.open(item["input"]).convert("RGB")
            image_1 = Image.open(item["edited"]).convert("RGB")
        except (PIL.UnidentifiedImageError, FileNotFoundError):
            print(f"\n {item['input']} or {item['edited']} is not a valid image")
            i = random.randint(0, len(self.meta) - 1)
            return self.__getitem__(i)
        prompt = item["instruction"]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), RESAMPLING_METHOD)
        image_1 = image_1.resize((reize_res, reize_res), RESAMPLING_METHOD)
        if self.instruct:
            prompt = "Image Editing: " + prompt
        # return image_0, image_1, prompt

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))