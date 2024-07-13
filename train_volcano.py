# from utils.llava_flash_attn import replace_llama_attn_with_flash_attn
import os, random
from tkinter import FALSE
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
import torch
from lightning.pytorch import LightningModule, seed_everything
from transformers import Trainer, TrainerCallback
from vocot_trainer import VoCoTTrainer
from transformers.trainer_callback import TrainerControl, TrainerState
import transformers
from torch.utils.data import ConcatDataset
from typing import Optional, Dict
from dataclasses import dataclass, field
from locals.datasets import SFT_DataCollator
import logging, re, shutil
from pathlib import Path
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from transformers import CLIPImageProcessor

from lightning.pytorch.callbacks import BasePredictionWriter
from locals.datasets.preprocessor import VoCoT_InputProcessor
import argparse
from omegaconf import OmegaConf
from utils.util import instantiate_from_config, print_trainable_params, safe_save_model_for_hf_trainer
import torch.distributed as dist
import pathlib
from PIL import Image
from model.language_model.volcano_llama import VolCanoLlamaForCausalLM, VolCanoConfig
from model.language_model.volcano_mistral import VolCanoMistralForCausalLM, VolCanoMistralConfig
from transformers import LlamaTokenizer, AutoTokenizer
import json 
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class PredWriter(BasePredictionWriter):
    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Any,  # complex variables is ok
        batch_indices: list[list[list[int]]],
    ) -> None:
        output_folder = pl_module.output_folder
        torch.save(predictions, os.path.join(output_folder, f"predictions-{trainer.local_rank}.pt"))
        rank0_print(f'rank {trainer.local_rank} predictions saved')

def default_gpus():
    return [0,1,2,3]

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="multimodal_encoder")
    model_path: Optional[str] = field(default="multimodal_encoder")
    flash_attn: Optional[bool] = field(default=False)
    snr_loss: Optional[bool] = field(default=True)
    use_mistral: Optional[bool] = field(default=False)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    stage1_weight: Optional[str] = field(default=None)
    stage2_weight: Optional[str] = field(default=None)
    sd_base_name: Optional[str] = field(default="stabilityai/stable-diffusion-2-1-base")
    vision_encoder: Optional[str] = field(default="/mnt/bn/yangmin-priv/luoruipu/weight_pretrained/eva_vit_g.pth")
    vision_encoder_path: Optional[str] = field(default=None)
    front_projector_type: Optional[str] = field(default="q_former")
    front_projector: Optional[str] = field(default="/mnt/bn/yangmin-priv/luoruipu/weight_pretrained/blip2_pretrained_flant5xxl.pth")
    num_query_token: Optional[int] = field(default=32)
    front_projector_type: Optional[str] = field(default="q_former")
    vision_generator_type: Optional[str] = field(default='SD')
    behind_projector: Optional[str] = field(default='linear')
    vision_generator_cond_channels: Optional[int] = field(default=4)
    vision_generator: Optional[str] = field(default = "/mnt/bn/luoruipu-disk/weight_pretrained/stable-diffusion-2-1-base/")
    behind_projector: Optional[bool] = field(default = True)
    t2i_mapping_hidden_size: Optional[int] = field(default=1024)
    compute_diffusion_loss: Optional[bool] = field(default=False)
    avoid_generator: Optional[bool] = field(default=False)
    tokenizer_model_max_length: Optional[int] = field(default=None)
    reinit_embedding_size: Optional[int] = field(default=None)
    num_image_token: Optional[int] = field(default=64)
    extend_loc_vocabulary: Optional[bool] = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    data_config_path: str = field(default=None, metadata={"help": "Path to the data config file."})
    expand_to_square: bool = field(default=False, metadata={"help": "Whether to expand the image into square before resize"})
    project_name: str = field(default='edit_minigpt5')
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    lora_enable: bool = field(default = True)
    lora_save_strategy: str = field(default = None)
    stage1_ckpt: str = field(default=None)
    loss_weight_decay: str = field(default='none')
    regression_weight: Optional[float] = field(default=1.0)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_backbone: bool = field(default=False)
    freeze_vision_generator: bool = field(default=True)
    output_dir: str = field(default=WEIGHTFOLDER)
    num_train_epochs:int = field(default=2)
    per_device_train_batch_size:int = field(default=2)
    per_device_eval_batch_size:int = field(default=2)
    real_batch_size:int = field(default=48)
    save_total_limit:int = field(default=1)
    learning_rate:float = field(default=2e-5)
    warmup_ratio:float = field(default=0.03)
    # warmup_steps:int = field(default=1000)
    adam_epsilon:float = field(default=1e-8)
    deepspeed: str = field(default=None)
    stage: int = field(default=2)
    num_workers:int = field(default=4)
    activate_behind_fc: bool = field(default=False)
    activate_behind_projector: bool = field(default=True)
    activate_behind_query: bool = field(default=False)
    model_max_length: int = field(default=512)

    gpus: List[int] = field(default_factory=default_gpus)
    resume: Optional[str] = field(default=None)
    is_training: Optional[bool] = field(default=False)
    test_weight: Optional[str] = field(default=None)
    lora_weight: Optional[str] = field(default=None)
    skip_vision_encoder_load: Optional[bool] = field(default=False)
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    
    def output_log(self, args: TrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_enable and args.lora_save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            self.output_log(args, state)
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            state_dict = get_peft_state_maybe_zero_3(
                model_.named_parameters(), 'none'
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model_.named_parameters()
            )
            if args.local_rank == 0 or args.local_rank == -1:
                output_dir = os.path.join(args.output_dir,f'checkpoint-{save_number}')
                os.makedirs(output_dir, exist_ok=True)
                # model_.config.save_pretrained(output_dir)
                model_.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
                kwargs["tokenizer"].save_pretrained(output_dir)
        elif args.save_strategy == 'steps' and (state.global_step - 1)%args.save_steps == 0:
            # no need to manual saving
            ordering_and_checkpoint_path = []
            checkpoint_prefix = 'checkpoint'
            glob_checkpoints = [str(x) for x in Path(args.output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            # checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
            if len(checkpoints_sorted) > 1:
                # there are more than 2 checkpoints, delete the one before the latest
                checkpoint_to_delete = os.path.join(checkpoints_sorted[-2][1], 'global_step{}'.format(checkpoints_sorted[-2][0]))
                rank0_print('deleting checkpoint_to_delete for saving memory')
                shutil.rmtree(checkpoint_to_delete, ignore_errors=True)

        elif args.save_strategy == 'no' and args.lora_save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            # if not using huggingface save, perform manual saving
            self.output_log(args, state)
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            state_dict_to_store = get_peft_state_non_lora_maybe_zero_3(model_.named_parameters())
            if args.local_rank == 0 or args.local_rank == -1:
                output_dir = os.path.join(args.output_dir,f'checkpoint-{save_number}')
                os.makedirs(output_dir, exist_ok=True)
                # model_.config.save_pretrained(output_dir)
                torch.save(state_dict_to_store, os.path.join(output_dir, 'pytorch_model.bin'))
                kwargs["tokenizer"].save_pretrained(output_dir)
        # perform the weight
        if args.loss_weight_decay == 'linear':
            max_steps = state.max_steps
            global_step = state.global_step
            init_weight = args.regression_weight
            weight_per_step = (init_weight - 1) / max_steps
            current_weight = init_weight - (weight_per_step * global_step)
            model_ = kwargs["model"]
            model_.regression_weight = 2 * current_weight / (1 + current_weight)
        elif args.loss_weight_decay == 'none':
            pass
        else:
            raise NotImplementedError
        return super().on_step_end(args, state, control, **kwargs)

def change_trainable_params(training_args, model_args, model):
    if training_args.stage == 1:
        raise NotImplementedError
    elif training_args.stage == 2:
        if model_args.vision_generator_type == 'P2P_SD' and not training_args.freeze_vision_generator:
            for k,v in model.named_parameters():
                if 'unet' in k:
                    v.requires_grad=True
        for k,v in model.named_parameters():
                if 'behind_projector' in k and 't2i_decoder_prompt' not in k and training_args.activate_behind_projector:
                    v.requires_grad = True
                if training_args.activate_behind_query:
                    if 'behind_projector.t2i_decoder_prompt' in k:
                        v.requires_grad = True
                if training_args.activate_behind_fc:
                    if 'model.fc' in k:
                        v.requires_grad = True
                elif 'lora' in k:
                    v.requires_grad = True

class CLIPTransform:
    def __init__(self, transform):
        self.transform = transform
        self.image_mean = transform.image_mean
    
    def __call__(self, image):
        width, height = image.size
        if width == 1 and height == 1:
            # (1,1) image
            image = image.resize((16, 16))
        try:
            rep = torch.tensor(self.transform(image)['pixel_values'][0])
        except:
            rep = torch.tensor(self.transform(Image.new(image.mode, (32, 32), (0,0,0)))['pixel_values'][0])
        return rep

def llava_projector_mapping(ckpt):
    key_pairs = []
    new_dict = {}
    for k,v in ckpt.items():
        if k.startswith('model.mm_projector'):
            new_k = k.replace('model.mm_projector', 'front_mm_projector')
        else:
            new_k = 'front_mm_projector.'+k
        key_pairs.append([k, new_k])
    for pair in key_pairs:
        k, new_k = pair
        new_dict[new_k] = ckpt[k]
        del(ckpt[k])
    return new_dict

def main(args):
    # replace_llama_attn_with_flash_attn()
    global local_rank
    os.environ['NCCL_DEBUG']=''
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf,allow_extra_keys=True)
    local_rank = training_args.local_rank
    training_args.learning_rate = float(training_args.learning_rate) # fix the learning rate type
    os.environ['WANDB_PROJECT'] = data_args.project_name
    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]

    # distinguish mistral and llama backbone
    if model_args.use_mistral:
        print('based on Mistral model')
        config_class = VolCanoMistralConfig
        model_class = VolCanoMistralForCausalLM
        tokenizer_class = AutoTokenizer
    else:
        print('based on Llama model')
        config_class = VolCanoConfig
        model_class = VolCanoLlamaForCausalLM
        tokenizer_class = LlamaTokenizer

    # model construction
    llama_config = config_class.from_pretrained(model_args.model_path)
    if model_args.flash_attn:
        print('using flash attn!')
        llama_config._flash_attn_2_enabled = True
        llama_config._attn_implementation = 'flash_attention_2'
    else:
        print('run without flash attn')
    llama_config.skip_load_vision_encoder = training_args.skip_vision_encoder_load
    llama_config.num_image_token = model_args.num_image_token
    if training_args.bf16:
        current_dtype = torch.bfloat16
    else:
        current_dtype = torch.float16
    model = model_class.from_pretrained(model_args.model_path, config=llama_config, torch_dtype=current_dtype)
    
    
    if model_args.use_mistral:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=True,
            trust_remote_code=True
        )
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_path, use_fast=False)
    model.tokenizer = tokenizer
    model.config.vision_encoder = model_args.vision_encoder
    model.config.vision_encoder_path = model_args.vision_encoder_path
    model.config.front_projector_type = model_args.front_projector_type
    model.config.front_projector = model_args.front_projector
    model.config.num_query_token = model_args.num_query_token
    model.config.vision_generator = model_args.vision_generator
    model.config.vision_generator_type = model_args.vision_generator_type
    model.config.behind_projector = model_args.behind_projector
    model.config.t2i_mapping_hidden_size = model_args.t2i_mapping_hidden_size
    
    model.config.behind_projector = model_args.behind_projector
    model.config.snr_loss = True
    model.config.compute_diffusion_loss = model_args.compute_diffusion_loss
    model.config.avoid_generator = model_args.avoid_generator
    model.config.tokenizer_model_max_length = model_args.tokenizer_model_max_length
    model.config.num_image_token = model_args.num_image_token
    model.config.mm_vision_select_layer = model_args.mm_vision_select_layer

    if not hasattr(model, 'vision_encoder'):
        model.init_vision_model()
    model.regression_weight = 2 * training_args.regression_weight / (1 + training_args.regression_weight)
    
    model.init_tokenizer_grd(tokenizer)
    if model_args.use_mistral:
        tokenizer.pad_token = tokenizer.unk_token
    if training_args.stage1_ckpt:
        print('loading the stage 1 checkpoint!')
        stage1_ckpt = torch.load(training_args.stage1_ckpt, map_location='cpu')
        stage1_ckpt = llava_projector_mapping(stage1_ckpt)
        base_dir = os.path.dirname(training_args.stage1_ckpt)
        if os.path.exists(os.path.join(base_dir, 'tokenizer_config.json')):
            tokenizer = tokenizer_class.from_pretrained(base_dir, use_fast=False)
            model.tokenizer = tokenizer
            model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(stage1_ckpt, strict=False)
    if model_args.extend_loc_vocabulary:
        model.init_tokenizer_loc(tokenizer)

    if model_args.reinit_embedding_size is not None:
        model.reinit_partial_embeddings(model_args.reinit_embedding_size)

    # tokenizer.pad_token = tokenizer.eos_token
    print('length tokenizer',len(tokenizer))
    if training_args.freeze_backbone:
        model.model.requires_grad_(False)


    if training_args.lora_enable:
        print("Using LoRA")
        # lora_target_modules = [f"model.layers.{i}.self_attn.q_proj" for i in range(32)] + [f"model.layers.{i}.self_attn.v_proj" for i in range(32)] + [f"model.layers.{i}.self_attn.k_proj" for i in range(32)] 
        avoid_keys = ['embed_tokens', 'lm_head', 'front_mm_projector', 'behind_projector', 'vision_encoder', 'llama_proj', 'vision_generator']
        lora_target_modules = []
        for k,v in model.named_modules():
            if any(mm_keyword in k for mm_keyword in avoid_keys):
                continue
            elif isinstance(v, torch.nn.Linear):
                lora_target_modules.append(k)
        # lora_target_modules = [k for k,v in model.named_modules if isinstance(v, torch.nn.Linear)]
        # if model_args.vision_generator_type == 'P2P_SD' and training_args.stage == 2:
        #     lora_target_modules = lora_target_modules + ['unet']
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=['lm_head','embed_tokens', 'front_mm_projector', 'behind_projector']
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        model = get_peft_model(model, lora_config)
        model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
        model.base_model.model.lm_head.original_module.weight.requires_grad = False

    else:
        # freeze the vision encoder
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        # activate the input embeddings and lm head
        model.get_input_embeddings().weight.requires_grad = True

    sd_tokenizer = None

    # data create

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
                transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                # transforms.RandomHorizontalFlip(), comment here
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
    if hasattr(model.vision_encoder, 'image_processor'):
        input_vis_processor = model.vision_encoder.image_processor
        input_vis_processor = CLIPTransform(input_vis_processor)
    preprocessor = VoCoT_InputProcessor(tokenizer=tokenizer, input_image_processor=input_vis_processor, use_mistral=model_args.use_mistral,
                                                output_image_processor=output_vis_processor, merge_in_out_image=True, expand2square=data_args.expand_to_square)

    data_collator = SFT_DataCollator(tokenizer=tokenizer, sd_tokenizer=None)
    # make the dataloader here!
    config = OmegaConf.load(data_args.data_config_path)
    ds_helper = instantiate_from_config(config['datasets'])
    ds_helper.wrap = True
    ds_helper.preprocessor = preprocessor
    ds_helper.setup()
    for k,v in ds_helper.datasets.items():
        if hasattr(v.data, 'expand2square') and v.data.expand2square != data_args.expand_to_square:
            print('error with {} dataset'.format(k))
            v.data.expand2square = data_args.expand_to_square
    concate_ds = ConcatDataset([v for v in ds_helper.datasets.values()])

    
    if local_rank == 0:
        print('test sample')
        print(concate_ds[random.randint(0, len(concate_ds))])

    # from torch.utils.data import DataLoader
    # sampler_train = torch.utils.data.DistributedSampler(
    #         concate_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    #     )
    # dl = DataLoader(concate_ds, batch_size=training_args.per_device_train_batch_size, sampler=sampler_train)

    change_trainable_params(training_args, model_args, model)


    callback_class = LLMCallback
    trainer = VoCoTTrainer(model = model,
                        tokenizer = tokenizer,
                        args = training_args,
                        callbacks=[callback_class],
                        train_dataset=concate_ds,
                        data_collator=data_collator,
                        eval_dataset=None)
    # trainer._signature_columns = ['input_images','output_images']

    print_trainable_params(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # Lora model is not support this resume branch, make sure your lora out_dir is empty.
        rank0_print('resume')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()

    model.config.use_cache = True

    # delelet the last checkpoint running states
    if training_args.save_strategy == 'steps':
        # no need to manual saving
        ordering_and_checkpoint_path = []
        checkpoint_prefix = 'checkpoint'
        glob_checkpoints = [str(x) for x in Path(training_args.output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        # checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        if len(checkpoints_sorted) > 1:
            # there are more than 2 checkpoints, delete the one before the latest
            checkpoint_to_delete = os.path.join(checkpoints_sorted[-1][1], 'global_step{}'.format(checkpoints_sorted[-1][0]))
            rank0_print('deleting checkpoint_to_delete for saving memory')
            shutil.rmtree(checkpoint_to_delete, ignore_errors=True)


    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), 'none'
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,
                        default="valley/configs/experiment/valley_debug.yaml")
    args = parser.parse_args()
    main(args)