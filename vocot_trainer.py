import copy
import json
import os
import math
import numpy as np
from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from torch import nn
import datasets
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, SequentialSampler, DistributedSampler
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers import Trainer, is_datasets_available, PreTrainedModel
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction,seed_worker
import torch
import re
from transformers.data.data_collator import DataCollator


class VoCoTTrainer(Trainer):
    def __init__(self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        regression_text_loss_metrics: Optional[Tuple[float]]=(0.0, 0.0)):
        self.regression_text_loss_metrics = regression_text_loss_metrics

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics,)
    
    def log(self, logs: Dict[str, float], eval = False) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if not eval:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

            # logs['lora_lr'] = self.optimizer.param_groups[0]['lr']
            # logs['other_lr'] = self.optimizer.param_groups[1]['lr']
            txt_loss, reg_loss = self.regression_text_loss_metrics
            if txt_loss is not None:
                logs['text_loss'] = txt_loss
            if reg_loss is not None:
                logs['regression_loss'] = reg_loss
            output = {**logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        else:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # def create_optimizer(self,):
    #     if self.args.lora and self.args.tune_mm_mlp_adapter:
    #         from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    #         from transformers.trainer_pt_utils import get_parameter_names
    #         decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
    #         decay_parameters = [name for name in decay_parameters if "bias" not in name]
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [
    #                     p for n, p in self.model.named_parameters() if ('lora' in n and p.requires_grad)
    #                 ],
    #                 "weight_decay": self.args.weight_decay,
    #                 'lr': float(self.args.lora_lr) if self.args.lora_lr else self.args.learning_rate
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in self.model.named_parameters() if (n in decay_parameters and 'lora' not in n and p.requires_grad)
    #                 ],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
    #                 ],
    #                 "weight_decay": 0.0,
    #             },
    #         ]
    #         optimizer_cls, optimizer_kwargs = ValleyTrainer.get_optimizer_cls_and_kwargs(self.args)
    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #     else:
    #         self.optimizer = super().create_optimizer()
    #     return self.optimizer
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        if 'gt_label' in inputs:
            gt_label = inputs.pop('gt_label')
        return super()._prepare_inputs(inputs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # return super().compute_loss(model, inputs, return_outputs=False)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # record the regression and text loss
        text_loss = outputs['text_loss']
        if 'regression_loss' in outputs:
            regression_loss = outputs['regression_loss']
        else:
            regression_loss = None
        if self.args.n_gpu > 1:
            text_loss = text_loss.mean()
            if regression_loss is not None:
                regression_loss = regression_loss.mean()
        text_loss = text_loss.item()
        if regression_loss is not None:
            regression_loss = regression_loss.item()
        else:
            regression_loss = self.regression_text_loss_metrics[1]
        self.regression_text_loss_metrics = (text_loss, regression_loss)
        return loss
    