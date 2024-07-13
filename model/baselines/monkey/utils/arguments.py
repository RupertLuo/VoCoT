from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import transformers
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="model_zoo/monkey",
        metadata={"help": "Path to pre-trained model."}
    )
    eval_model_path: Optional[str] = field(
        default="checkpoints/monkey_cache",
        metadata={"help": "Path to local finetuned model."}
    )


@dataclass
class DataArguments:
    experiment_name: str="unknown"
    train_data_path: str = field(
        default="data/aitw_with_gpt/train", 
        metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the evaluation data."}
    )
    test_data_path: str = field(
        default="data/aitw_with_gpt/test", 
        metadata={"help": "Path to the testing data."}
    )
    max_input_length:  int = 1792
    max_output_length: int = 256
    use_image_path:     bool = False
    use_history_action: bool = False
    use_history_image:  bool = False     # monkey do not support multiple images
    max_history_length: int = field(
        default=2, 
        metadata={"help": "Maximum history information used in GUI."}
    )
    use_dynamic_resolution: bool=False
    use_screen_desc: bool=False
    use_post_think: bool=False
    use_pre_think: bool=True
    use_action_desc: bool=True
    use_ground_operation: bool=False
    use_task_prompt: bool=False


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: str = field(default="checkpoints/monkey_log")
    cache_dir: Optional[str] = field(default="checkpoints/monkey_cache")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. " + \
                          "Sequences will be right padded (and possibly truncated)."},
    )
    zero_stage: int = 2
    use_lora: bool = True
    use_peft: bool = False
    fix_vit:  bool = True
    remove_unused_columns: bool=False
    per_device_train_batch_size: int=1
    per_device_eval_batch_size: int=4
    result_output_dir: str=None


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["in_proj","out_proj","c_fc"] 
        ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False



def get_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args, 
        data_args, 
        training_args, 
        lora_args
    ) = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, lora_args

