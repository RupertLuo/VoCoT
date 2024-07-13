import os
import random
import copy
import json
import math
from pathlib import Path
from tkinter import E
from typing import Any

import numpy as np
import torch
import torchvision
import pandas as pd
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from utils.util import byte2image
from ..utils.box_utils import *
from datasets import load_dataset

from constants import * 

class CoTCollectionDataset(Dataset):
    def __init__(self,
        path:str = None,
        sample_weight: float=1.0,
    ):
        self.meta = load_dataset(path)['train']
        self.sample_weight = sample_weight

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        item = self.meta[i]
        question = item['source'] + ' Answer the question and include the reasoning proess.'
        thought = item['rationale']
        answer = item['target']

        conv = [{'from': 'human', 'value': question},
                {'from': 'gpt', 'value': thought},
                {'from': 'human', 'value': 'What is your final answer?'},
                {'from': 'gpt', 'value': answer}]
        return {'conversation': conv}