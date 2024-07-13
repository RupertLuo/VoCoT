from genericpath import samestat
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from locals.datasets import SFT_DataCollator, WrappedDataset
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from locals.datasets.preprocessor import VoCoT_InputProcessor
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from model.language_model.volcano_llama import VolCanoLlamaForCausalLM,VolCanoConfig
from model.language_model.volcano_mistral import VolCanoMistralForCausalLM, VolCanoMistralConfig
from transformers import LlamaTokenizer, AutoTokenizer
import transformers
from peft import PeftConfig, PeftModel
from argparse import ArgumentParser
import os
import torch.distributed as dist
from utils.logger import setup_logger
import json
import tqdm

def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]

class CLIPTransform:
    def __init__(self, transform, square_size=None):
        self.transform = transform
        self.square_size = square_size
        self.image_mean = transform.image_mean
    
    def __call__(self, image):
        if self.square_size is not None:
            image = image.resize((self.square_size, self.square_size))
        try:
            tmp = torch.tensor(self.transform(image)['pixel_values'][0])
        except:
            tmp = torch.tensor(self.transform(Image.new(image.mode, (32, 32), (0,0,0)))['pixel_values'][0])
        return tmp



def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/mnt/bn/yangmin-priv/luoruipu/checkpoints/Edit-gpt-4-emu-instruct-test/')
    parser.add_argument('--eval_data', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--avoid_image_gen', action='store_true')
    parser.add_argument('--for_llava', action='store_true')
    parser.add_argument('--expand2square', action='store_true')
    parser.add_argument('--resize2square', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--sub_sample', type=int, default=-1)
    parser.add_argument('--evaluate_loss', action='store_true')
    parser.add_argument('--option_instruct', action='store_true')
    parser.add_argument('--no_barrier', action='store_true')
    parser.add_argument('--use_mistral', action='store_true')
    parser.add_argument('--likelihood_reduction', type=str, default='mean')
    parser.add_argument('--txt_prompt', action='store_true')
    parser.add_argument('--desc', action='store_true')
    parser.add_argument('--no_bind', action='store_true')
    parser.add_argument('--sub_image', action='store_true')
    args = parser.parse_args()

    print(args)

    base_config_name = os.path.basename(args.eval_data)
    args.dataset_name = base_config_name[:-5] if base_config_name.endswith('.yaml') else base_config_name
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.n_gpus = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.distributed = False
        args.local_rank = -1
        args.n_gpus = -1
    if not os.path.isdir(args.output_dir) and args.local_rank < 1:
        os.makedirs(args.output_dir)
    global logger
    logger = setup_logger('Evaluation', args.output_dir, args.local_rank)
    logger.info('Evaluating with {} GPUs'.format(args.n_gpus))

    if os.path.exists(get_output_name(args, mid_output=False)):
        print('the results already exist, finished!')
        return

    model_path = args.model_path
# parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
# conf = 'config/experiments/debug/stage2_debug.yaml'
# model_args, data_args, training_args = parser.parse_yaml_file(conf,allow_extra_keys=True)
# # model = EditGPTLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    # distribute the mistral and vicuna-based models
    if args.use_mistral:
        logger.info('based on Mistral model')
        config_class = VolCanoMistralConfig
        model_class = VolCanoMistralForCausalLM
        tokenizer_class = AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            use_fast=True,
            trust_remote_code=True
        )
    else:
        logger.info('based on Llama model')
        config_class = VolCanoConfig
        model_class = VolCanoLlamaForCausalLM
        tokenizer_class = LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    logger.info('loading model from {}'.format(model_path))
    llama_config = config_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=llama_config)

    model.input_img_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMG_TOKEN)
    model.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
    model.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    model.tokenizer = tokenizer
    model.no_bind = args.no_bind
    model.sub_image_bind = args.sub_image
    if not args.avoid_image_gen:
        model.config.avoid_generator = False
        model.init_vision_generator()

    if args.precision == 'bf16':
        model.to(torch.bfloat16)
    elif args.precision == 'fp16':
        model.to(torch.float16)
    model.eval()
    model.to(device)

    output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    # transforms.RandomHorizontalFlip(), # comment here
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    input_vis_processor = transforms.Compose(
            [
                transforms.Resize((448, 448) if args.resize2square else 448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                # transforms.RandomHorizontalFlip(), comment here
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        )
    if hasattr(model.vision_encoder, 'image_processor'):
        input_vis_processor = model.vision_encoder.image_processor
        if args.resize2square:
            tmp_size = input_vis_processor.size['shortest_edge']
        else:
            tmp_size = None
        input_vis_processor = CLIPTransform(input_vis_processor, square_size=tmp_size)
    # tokenizer = LlamaTokenizer.from_pretrained('eval/debug/edit_gpt_emu_tokenizer')
    if args.sub_image:
        model.image_processor = input_vis_processor
    else:
        model.image_processor = None
    preprocessor = VoCoT_InputProcessor(tokenizer=tokenizer, input_image_processor = input_vis_processor, use_mistral=args.use_mistral,
                                                output_image_processor= output_vis_processor, merge_in_out_image=True, expand2square=args.expand2square, inference = True)

    data_collator = SFT_DataCollator(tokenizer=tokenizer, sd_tokenizer=None)
    # make the dataloader here!
    config = OmegaConf.load(args.eval_data)
    logger.info('loading dataset as {}'.format(json.dumps(OmegaConf.to_object(config))))
    dataset = instantiate_from_config(config[0])
    if args.sub_sample > 0:
        dataset.meta = dataset.meta[:args.sub_sample]
    if args.for_llava:
        dataset.for_llava = True
    if hasattr(dataset, 'expand2square'):
        if dataset.expand2square != args.expand2square:
            print('unmatched expand2square parameter, please check it')
            dataset.expand2square = args.expand2square
    if args.txt_prompt:
        # using text COT prompt
        assert hasattr(dataset, 'txt_prompt'), 'current dataset does not support txt CoT prompt'
        dataset.txt_prompt = args.txt_prompt
    # activate the CoT
    if args.cot:
        if not hasattr(dataset, 'require_cot'):
            print('the current benchmark does not support COT!')
            raise ValueError
        else:
            dataset.require_cot = args.cot
    if args.local_rank <= 0:
        import random
        print('example evaluation data')
        print(dataset[random.randint(0, len(dataset)-1)])
    wrapped_dataset = WrappedDataset('eval', dataset, preprocessor)
    sampler = SequentialSampler(dataset) if not args.distributed else DistributedSampler(dataset, shuffle=False)
    dl = DataLoader(wrapped_dataset, sampler=sampler, batch_size=1, collate_fn=data_collator)
    logger.info("***** Runing Evaluation *****")
    logger.info("  Num examples = %d", len(dataset))

    current_res = []
    with torch.inference_mode(), torch.no_grad():
        for batch in tqdm.tqdm(dl, desc='evaluating'):
            if args.evaluate_loss:
                assert args.likelihood_reduction in ['sum', 'mean']
                # pred, t = model.calculate_options(batch, cot=args.cot, max_new_tokens=args.max_new_tokens, temperature=args.temperature, further_instruct=args.option_instruct)
                try:
                    pred, t = model.calculate_options(batch, cot=args.cot, max_new_tokens=args.max_new_tokens, temperature=args.temperature, further_instruct=args.option_instruct, likelihood_reduction=args.likelihood_reduction)
                except:
                    print('fail to predict {}'.format(batch['item_id'][0]))
                    pred = ''
                    t = ''
                tmp_dict = {
                        'item_id': batch['item_id'][0],
                        'predict': pred
                    }
                if args.cot:
                    tmp_dict['thought'] = t
                if hasattr(dataset, 'getlabel'):
                    try:
                        item_id = int(tmp_dict['item_id'].split('_')[-1])
                        tmp_dict['label'] = dataset.getlabel(item_id)
                    except:
                        pass
                current_res.append(tmp_dict)
                continue
            # generation-based evaluation here
            # txt_res, out_imgs, txt_ids = model.condition_completion(batch, avoid_image_gen=args.avoid_image_gen, 
            #                                                 max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            # item_id = int(batch['item_id'][0].split('_')[-1])
            # final_round_input = dataset.cot_turn(item_id, txt_res[0], txt_ids, eoc_id=model.eoc_token_id, img_id=model.input_img_id, mistral=args.use_mistral, sub_image_bind=args.sub_image)
            # final_batch = data_collator([preprocessor(final_round_input)])
            # final_pred, out_imgs, _ = model.condition_completion(final_batch, avoid_image_gen=args.avoid_image_gen,
            #                                                 max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            # txt_res, out_imgs, txt_ids = model.condition_completion(batch, avoid_image_gen=args.avoid_image_gen, 
            #                                                 max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            # if args.cot and not args.desc:
            #     item_id = int(batch['item_id'][0].split('_')[-1])
            #     final_round_input = dataset.cot_turn(item_id, txt_res[0], txt_ids, eoc_id=model.eoc_token_id, img_id=model.input_img_id, mistral=args.use_mistral, sub_image_bind=args.sub_image)
            #     final_batch = data_collator([preprocessor(final_round_input)])
            #     final_pred, out_imgs, _ = model.condition_completion(final_batch, avoid_image_gen=args.avoid_image_gen,
            #                                                     max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            #     tmp_dict = {
            #         'item_id': batch['item_id'][0],
            #         'thought': txt_res[0],
            #         'prediction': final_pred[0]
            #     }
            try:
                txt_res, out_imgs, txt_ids = model.condition_completion(batch, avoid_image_gen=args.avoid_image_gen, 
                                                            max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                if args.cot and not args.desc:
                    item_id = int(batch['item_id'][0].split('_')[-1])
                    final_round_input = dataset.cot_turn(item_id, txt_res[0], txt_ids, eoc_id=model.eoc_token_id, img_id=model.input_img_id, mistral=args.use_mistral, sub_image_bind=args.sub_image)
                    final_batch = data_collator([preprocessor(final_round_input)])
                    final_pred, out_imgs, _ = model.condition_completion(final_batch, avoid_image_gen=args.avoid_image_gen,
                                                                    max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                    tmp_dict = {
                        'item_id': batch['item_id'][0],
                        'thought': txt_res[0],
                        'prediction': final_pred[0]
                    }
                else:
                    tmp_dict = {
                        'item_id': batch['item_id'][0],
                        'prediction': txt_res[0]
                    }
            except:
                print('fail to predict {}'.format(batch['item_id'][0]))
                if args.cot:
                    tmp_dict = {
                        'item_id': batch['item_id'][0],
                        'thought': '',
                        'prediction': ''
                    }
                else:
                    tmp_dict = {
                        'item_id': batch['item_id'][0],
                        'prediction': ''
                    }
            if hasattr(dataset, 'getlabel'):
                try:
                    item_id = int(tmp_dict['item_id'].split('_')[-1])
                    tmp_dict['label'] = dataset.getlabel(item_id)
                except:
                    pass
            if hasattr(dataset, 'get_index'):
                item_id = int(tmp_dict['item_id'].split('_')[-1])
                tmp_dict['dataset_id'] = dataset.get_index(item_id)
            current_res.append(tmp_dict)
    # remove duplication if necessary in Distributed version
    if args.distributed and len(dataset) % args.n_gpus != 0:
        residual_samples = len(dataset) % args.n_gpus
        if not args.local_rank < residual_samples:
            current_res = current_res[:-1]
    
    with open(get_output_name(args, mid_output=True), 'w') as wf:
        json.dump(current_res, wf)

    print('====Finished From Rank {}====='.format(args.local_rank))

    if args.no_barrier:
        torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
        return

    torch.distributed.barrier()
    if args.local_rank == 0 or args.local_rank == -1:
        full_res = []
        for fn in get_all_output_names(args):
            full_res.extend(json.load(open(fn, 'r')))
            os.remove(fn)
        with open(get_output_name(args, mid_output=False), 'w') as wf:
            json.dump(full_res, wf)
        # saving the arguments
        torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
            

if __name__=='__main__':
    main()