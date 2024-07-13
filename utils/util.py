
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import textwrap
import importlib
from prettytable import PrettyTable
import torch.distributed as dist
import transformers
import torch
from safetensors import safe_open
from PIL import Image
from base64 import b64encode, b64decode
import io

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_safetensor(path):
    tmp_dict = {}
    with safe_open(path, framework='pt', device=0) as f:
        for k in f.keys():
            tmp_dict[k] = f.get_tensor(k)
    return tmp_dict


def sanitize_filename(filename):
    return re.sub('[^0-9a-zA-Z]+', '_', filename)

def plot_images_and_text(predicted_image1, predicted_image2, groundtruth_image, generated_text, gt_text, save_dir, task_name, input_texts, input_images):
    task_path = os.path.join(save_dir, task_name)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    max_width = 50  # adjust this value based on your needs

    fig, ax = plt.subplots()
    ax.imshow(predicted_image1)
    generated_text = generated_text.replace("###", "").replace("[IMG0]", "")
    wrapped_generated_text = textwrap.fill(generated_text, max_width)
    ax.set_title(wrapped_generated_text, pad=20)
    ax.axis('off')
    plt.savefig(os.path.join(task_path, f"generated.jpg"), bbox_inches='tight')
    plt.close(fig)

    gt_text = gt_text.replace("$", "\$")
    wrapped_gt = textwrap.fill(gt_text, max_width)
    if predicted_image2 is not None:
        fig, ax = plt.subplots()
        ax.imshow(predicted_image2)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"sd_baseline.jpg"), bbox_inches='tight')
        plt.close(fig)

    if groundtruth_image is not None:
        fig, ax = plt.subplots()
        groundtruth_image = groundtruth_image.float().cpu().numpy().squeeze()
        groundtruth_image = np.transpose(groundtruth_image, (1, 2, 0))
        groundtruth_image = np.uint8(groundtruth_image*255)
        ax.imshow(groundtruth_image)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"gt.jpg"), bbox_inches='tight')
        plt.close(fig)

    if len(input_texts):
        max_width = 30
        length = len(input_texts)
        if length > 1:
            fig, ax = plt.subplots(1, length, figsize=(10*length, 10))
            for i in range(length):
                if i < len(input_images):
                    ax[i].imshow(input_images[i])
                    ax[i].set_title(textwrap.fill(input_texts[i], max_width), fontsize=28)
                    ax[i].axis('off')
                else:
                    ax[i].text(0.5, 0.5, textwrap.fill(input_texts[i], max_width), horizontalalignment='center', verticalalignment='center', fontsize=28)
                    ax[i].axis('off')
        else:
            fig, ax = plt.subplots()
            ax.imshow(input_images[0])
            ax.set_title(textwrap.fill(input_texts[0], max_width), fontsize=28)
            ax.axis('off')
        plt.savefig(os.path.join(task_path, f"input.jpg"), bbox_inches='tight')
        plt.close(fig)

    return None

def instantiate_from_config(config, inference = False, reload=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"], reload=reload)(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def print_trainable_params(model):
    if dist.get_rank() == 0:
        trainable_params = [k for k,v in model.named_parameters() if v.requires_grad]
        trainable_params_group = {}
        for para in trainable_params:
            layer_num = re.findall(r'layers.(\d+)\.',para)
            if layer_num:
                cur_layer = int(layer_num[0])
                if para.replace('layers.'+layer_num[0],'layers.*') not in trainable_params_group:
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
                elif cur_layer > int(trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')]):
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
                    
            else:
                trainable_params_group[para] = '0'
        table = PrettyTable(['Parameter Name','Max Layer Number'])
        for key in trainable_params_group.keys():
            table.add_row([key, str(int(trainable_params_group[key])+1)])
        
        print(table)
        total_num = sum([v.numel() for k,v in model.named_parameters()])
        trainable_num = sum([v.numel() for k,v in model.named_parameters() if v.requires_grad])
        print('Total: {:.2f}M'.format(total_num/1e6), ' Trainable: {:.2f}M'.format(trainable_num/1e6))

def rank_0_print(output):
    if dist.get_rank() == 0:
        print(output)

def safe_save_model_for_hf_trainer(trainer,
                                   output_dir):
    """Collects the state dict and dump to disk."""
    
    if trainer.args.lora:
        if trainer.args.should_save: 
            trainer.model.save_pretrained(output_dir)
        
    else:
        if trainer.deepspeed:
            print('saving deepspeed model...')
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

def byte2image(byte_data):
    """
    convert byte to PIL image
    """
    if isinstance(byte_data, str):
        byte_data = b64decode(byte_data)
    image = Image.open(io.BytesIO(byte_data))
    return image
