import os
import time
import logging
import torch

import transformers
from transformers import deepspeed


def configure_logging():
    logger = logging.getLogger("gui")
    logger.setLevel(logging.INFO)
    if os.environ.get("LOGLEVEL", None) is not None:
        logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    
    if not logger.handlers: # stream handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        if os.environ.get("LOGLEVEL", None) is not None:
            console_handler.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        console_handler.setFormatter(formatter)
        # output to console
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


logger = configure_logging()

def print_rank0(msg, level=logging.INFO, flush=True):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if torch.distributed.is_initialized():
        msg = f"[RANK {torch.distributed.get_rank()}] {msg}"
        if torch.distributed.get_rank() == 0:
            logger.log(msg=msg, level=level)
            if flush: logger.handlers[0].flush()
    else:
        logger.log(msg=msg, level=level)



def safe_save_model(trainer: transformers.Trainer, output_dir: str):
    """ safe_save_model_for_hf_trainer.
        Collects the state dict and dump to disk. """
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)



def print_trainable_params(model: torch.nn.Module):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print_rank0(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}"
        .format(trainable_params, all_param, 100 * trainable_params / all_param))
