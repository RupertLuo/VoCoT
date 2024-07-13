from torch.utils.data import Dataset
import json as js
import math
import random
from tqdm import tqdm

class TextDataset(Dataset):

    def __init__(self, 
                 pathlist: list,
                 instruct: bool = False,
                 sample_weight: float = 1.0,
                 output_mode: str = 'text',
                 shuffle: bool = False,
                 inference: bool = False,
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.pathlist = pathlist
        self.instruct = instruct
        self.inference = inference
        self.meta = []
        for path in tqdm(pathlist):
            if path.endswith('json'):
                self.meta += js.load(open(path))
            elif path.endswith('jsonl'):
                with open(path) as f:
                    for line in f:
                        try:
                            self.meta.append(js.loads(line))
                        except:
                            continue
            else:
                raise ValueError('json or jsonl file type is supported. ')
        self.output_mode = output_mode
        self.shuffle = shuffle
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