from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


def get_image_transform(image_size:Union[int, Tuple[int, int]], mean:Optional[tuple]=None, std:Optional[tuple]=None):
    if mean is None:
        mean = (0.48145466, 0.4578275, 0.40821073)
    if std is None:
        std = (0.26862954, 0.26130258, 0.27577711)
    
    if type(image_size) is int:
        image_size = (image_size, image_size)
    
    image_transform = transforms.Compose([
            transforms.Resize(
                size=image_size,
                antialias=True,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return image_transform


'''======================================
               Padding Functions    
   ======================================'''


def padding_to_target(image:Image, target_size_hw:Tuple[int, int], mode:str):
    (w, h) = image.size
    (th, tw) = target_size_hw

    if mode == "upleft":
        padding = _get_upleft_pad(w, h, tw, th)
    elif mode == "central":
        padding = _get_central_pad(w, h, tw, th)
    else: raise NotImplementedError

    img_pad = F.pad(image, padding, fill=0, padding_mode="constant")
    return img_pad


def padding_to_square(image:Image, mode:str):
    (w, h) = image.size
    
    target_size = max(w, h)
    if mode == "upleft":
        padding = _get_upleft_pad(w, h, target_size, target_size)
    elif mode == "central":
        padding = _get_central_pad(w, h, target_size, target_size)
    else: raise NotImplementedError

    img_pad = F.pad(image, padding, fill=0, padding_mode="constant")
    return img_pad


def _get_central_pad(w:int, h:int, tw:int, th:int):
    h_padding = (tw - w) / 2
    v_padding = (th - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int (b_pad))
    return padding


def _get_upleft_pad(w:int, h:int, tw:int, th:int):
    l_pad, t_pad = 0, 0
    r_pad, b_pad = tw - w, th - h
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int (b_pad))
    return padding


'''======================================
            For Dynamic Resolution    
   ======================================'''

IMG_RES_TEMPLATES = [
    (896, 896), (448, 896), (896, 448), (448, 448)
] # (height, width)


def resolution_match(image:Image):
    (width, height) = image.size

    if max((width, height)) < 560: 
        RES_TEMPLATE, PAD = (448, 448), False
    
    if height == width:
        RES_TEMPLATE, PAD = (896, 896), False
    elif height > width:
        if height > 2.2 * width:
            RES_TEMPLATE, PAD = (896, 448), True
        elif height > 1.8 * width: 
            RES_TEMPLATE, PAD = (896, 448), False
        elif height > 1.2 * width:
            RES_TEMPLATE, PAD = (896, 896), True
        else:
            RES_TEMPLATE, PAD = (896, 896), False
    else:
        if width > 2.5 * height:
            RES_TEMPLATE, PAD = (448, 896), True
        elif width > 1.8 * height:
            RES_TEMPLATE, PAD = (448, 896), False
        elif width > 1.2 * height:
            RES_TEMPLATE, PAD = (896, 896), True
        else:
            RES_TEMPLATE, PAD = (896, 896), False
    
    PAD_SIZE = None
    if PAD:
        if RES_TEMPLATE in [(896, 896), (448, 448)]:
            PAD_SIZE = (max(width, height), max(width, height))
        elif RES_TEMPLATE == (896, 448):   # height > width
            PAD_SIZE = (height, height // 2)
        elif RES_TEMPLATE == (448, 896):   # height < width
            PAD_SIZE = (width // 2, width)
        else: raise NotImplementedError

    return RES_TEMPLATE, PAD, PAD_SIZE




'''======================================
              Hepler Functions    
   ======================================'''


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else: self._rank, self._world_size = 0, 1
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



if __name__ == "__main__":
    import numpy as np

    img = Image.open("/root/zjw_code/gui_nav/data/debug_samples/IMG_1739_resize.PNG")
    img = img.convert('RGB')

    (w, h) = img.size

    res_template, _pad, _pad_size_hw = resolution_match(img)
    print(f"Image shape: ({h}, {w}), Matching {res_template} (padding={_pad})")

    resize_transform = transforms.Resize(
        size=res_template,
        antialias=True,
        interpolation=InterpolationMode.BICUBIC
    )
    image_transform = get_image_transform(res_template)

    if _pad:
        img = padding_to_target(img, _pad_size_hw, mode="upleft")
        (new_w, new_h) = img.size
        print(f"Image shape: ({h}, {w}) -> Padding to ({new_h}, {new_w})")
    
        image_mask = torch.zeros((1, new_h, new_w))
        image_mask[:, :h, :w] = 1
        image_mask = resize_transform(image_mask).bool()
        image_mask = image_mask[0].unsqueeze(-1)

        image_tensor = torch.from_numpy(np.array(resize_transform(img)))
        masked_image = image_tensor * image_mask
        masked_image = masked_image.numpy()
        result_img = Image.fromarray(masked_image)
    else:
        image_array = np.array(resize_transform(img))
        result_img = Image.fromarray(image_array)

    result_img.save("/root/zjw_code/gui_nav/data/debug_samples/IMG_1739_processed.PNG")