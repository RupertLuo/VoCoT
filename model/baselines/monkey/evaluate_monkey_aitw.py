from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import json
import logging
import itertools

import random
import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader

from src_monkey.model import MonkeyConfig, MonkeyLMHeadModel, QWenTokenizer
from src_monkey.utils import get_args, print_rank0, print_trainable_params
from src_monkey.utils.data_utils import InferenceSampler
from src_monkey.dataset import AITWDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_collate_fn(features:List[dict], tokenizer:QWenTokenizer):
    step_ids = [ex['step_id'] for ex in features]
    
    input_ids = {'input_ids': [ex['input_ids'] for ex in features]}
    input_ids = tokenizer.pad(input_ids, padding='longest', return_tensors='pt')
    input_ids = input_ids['input_ids']
    attention_mask = (input_ids != tokenizer.pad_token_id)

    images = torch.stack([ex['images'] for ex in features], dim=0)          # B,C,H,W

    labels = tokenizer.batch_decode([ex['labels'] for ex in features], 
                                    skip_special_tokens=True, clean_up_tokenization_spaces=True)
    action_type = [ex['action_type'] for ex in features]
    action_desc = [ex['action_desc'] for ex in features]

    return step_ids, input_ids, attention_mask, \
           images, (labels, action_type, action_desc)


def do_evaluate(model, dataloader, save_name):
    print_rank0("[EVAL] Now start testing... ")

    outputs = []
    for _, (step_ids, input_ids, attention_msk, images, labels) \
        in enumerate(tqdm(dataloader)):

        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_msk.cuda(),
            images=images.cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256,
            min_new_tokens=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
        answers = [tokenizer.decode(p[input_ids.size(1):].cpu(), 
                                    skip_special_tokens=True).strip() for p in pred]
        
        labels, action_ts, action_ds = labels

        for step_id, answer, label, action_type, action_desc in \
            zip(step_ids, answers, labels, action_ts, action_ds):
            outputs.append({
                'step_id': step_id,
                'answer': answer,
                'label': label,
            })
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        time_prefix = time.strftime('%Y-%m-%d', time.localtime())
        results_file = os.path.join(training_args.result_output_dir, 
                                    f'aitw_{save_name}_{time_prefix}.json')
        json.dump(merged_outputs, 
                  open(results_file, 'w'), ensure_ascii=False, indent=4)

    torch.distributed.barrier()



if __name__ == "__main__":
    setup_seed(2024)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    model_args, data_args, training_args, lora_args = get_args()
    local_rank = training_args.local_rank

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print_rank0(f"[WORLD SIZE] {world_size}")

    ddp = (world_size != 1)
    device_map = None
    if not training_args.zero_stage == 3:
        if ddp and torch.cuda.is_available():
            device_map = f'cuda:{local_rank}'
        elif torch.cuda.is_available(): device_map = 'cuda'
        else: device_map = 'cpu'
    else:
        device_map = {'': local_rank} if ddp else None
    print_rank0(f"[Device] Current process using {device_map} device.")

    print_rank0('[Tokenizer] QWen Tokenizer loading... ')
    tokenizer = QWenTokenizer.from_pretrained(
        model_args.eval_model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = 'left'

    print_rank0('[Data] AITW dataset loading... ')
    test_set = AITWDataset(
        data_dir=data_args.test_data_path,
        tokenizer=tokenizer, args=data_args
    )
    print_rank0(f'[Data] Testing set has {len(test_set)} samples.')

    test_loader = DataLoader(
        dataset=test_set,
        sampler=InferenceSampler(len(test_set)),
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=world_size,
        pin_memory=True,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer)
    )

    print_rank0(f"[Model] Loading {model_args.eval_model_path}...")
    model = MonkeyLMHeadModel.from_pretrained(
        model_args.eval_model_path,
        device_map=device_map,
    ).eval()
    model.requires_grad_(False)
    print_trainable_params(model)
    
    do_evaluate(model, test_loader, data_args.experiment_name)
    print_rank0('===================================================')

    pass
