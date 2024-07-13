from utils.util import instantiate_from_config
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from functools import partial
import torch
import torch.distributed as dist
import transformers
from typing import Optional, Dict, Sequence
from constants import PRECISION, IGNORE_TOKEN_ID
from dataclasses import dataclass
import random

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, k, dataset, processor=None):
        self.ds_key = k
        self.data = dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.processor is not None:
            item = self.processor(self.data[idx])
        else:
            item = self.data[idx]
        item['item_id'] = '{}_{}'.format(self.ds_key, idx)
        return item
    
worker_init_fn = None

@dataclass
class SFT_DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sd_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        key_list = instances[0].keys()
        output_dict = {}
        # check if no regression loss
        # if "image_label_masks" in key_list:
        #     output_value = [instance["image_label_masks"] for instance in instances]
        #     if sum([k.sum().item() for k in output_value]) == 0:
        #         change_index = random.choice(list(range(len(instances))))
        #         change_item = instances[change_index]
        #         assert change_item['item_id'].startswith('cc3m') or change_item['item_id'].startswith('coco')
        #         change_item['image_label_masks'][0] = 1
        #         tmp_ids = change_item['input_ids']
        #         tmp_len = tmp_ids.shape[0]
        #         change_item['input_ids'] = torch.cat([tmp_ids[:1], tmp_ids[5:(tmp_len-1)], tmp_ids[1:5], tmp_ids[-1:]])
        #         change_item['labels'] = change_item['input_ids'].clone()
        for key in key_list:
            # Need to remove the batch dimension
            if key in ['input_ids', 'attention_mask', 'labels']:
                output_value = [instance[key] for instance in instances]
            else:
                output_value = [instance[key] for instance in instances]
            
            # processing based on different keys
            if key == "input_ids":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif key == "labels":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=IGNORE_TOKEN_ID)
            elif key == "attention_mask":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=0)
            elif key == 'input_images':
                tmp = []
                for v in output_value:
                    if v is not None:
                        tmp.append(v.to(PRECISION))
                    else:
                        tmp.append(v)
                output_value = tmp
            elif key == 'image_label_masks':
                tmp = []
                for v in output_value:
                    if v is not None:
                        tmp.append(v.to(PRECISION))
                    else:
                        tmp.append(v)
                output_value = tmp
            elif key == 'output_images':
                try:
                    output_value = torch.concat([item for item in output_value if item is not None], dim=0).to(PRECISION)
                except:
                    output_value = None
            elif key == 'output_cond_images':
                try:
                    output_value = torch.concat([item for item in output_value if item is not None], dim=0).to(PRECISION)
                except:
                    output_value = None
            elif key == 'output_cond_img_mask':
                try:
                    output_value = torch.concat([item for item in output_value if item is not None], dim=0).to(PRECISION)
                except:
                    output_value = None
            elif key == 'output_image_feature':
                output_value = torch.concat(output_value)
            elif key == 'box':
                tmp = []
                for v in output_value:
                    if v is not None:
                        tmp.append(v.to(PRECISION))
                    else:
                        tmp.append(v)
                output_value = tmp
            output_dict[key] = output_value
        # if 'labels' in output_dict and 'output_images' in output_dict:
        #     if output_dict['output_images'].shape[0] != len((output_dict['labels']==32000).nonzero().tolist()):
        #         raise ValueError
        return output_dict

class DataModuleFromConfig():
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, preprocessor=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.preprocessor = preprocessor
        if train is not None:
            if "target" in train:
                self.dataset_configs["train"] = train
                self.train_dataloader = self._train_dataloader
            else:
                for ds in train:
                    ds_name = str([key for key in ds.keys()][0])
                    self.dataset_configs[ds_name] = ds
                self.train_dataloader = self._train_concat_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(k, self.datasets[k], processor=self.preprocessor)

    def _train_concat_dataloader(self):
        is_iterable_dataset = True # isinstance(self.datasets['ds1'], Txt2ImgIterableBaseDataset)

        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        concat_dataset = []
        for ds in self.datasets.keys():
            concat_dataset.append(self.datasets[ds])

        concat_dataset = ConcatDataset(concat_dataset)
        sampler_train = torch.utils.data.DistributedSampler(
            concat_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(concat_dataset, batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

    def _train_dataloader(self):
        is_iterable_dataset = True # isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        sampler_train = torch.utils.data.DistributedSampler(
            self.datasets["train"], num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)
