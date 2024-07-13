from pathlib import WindowsPath
from constants import *
from utils.eval_util import extract_all_box_str

def process_thought(thought, mistral=False):
    new_thought = thought.replace(' <coor> ', '<coor>')
    # if ' </coor>  ' in new_thought:

    new_thought = new_thought.replace(' </coor> ', '</coor>' + ALL_IMG_TOKENS_STR).replace('</s>', '')
    all_box = extract_all_box_str(thought, mistral)
    return new_thought, all_box
    

def box2str(box, mode='special_tokens', prec=2, space=False):
    if mode == 'special_tokens':
        # using tokens to represent the locations
        num_blocks = len(ALL_LOC_TOKENS)
        size_per_block = 1 / num_blocks
        block_no = [int(c / size_per_block) if c < 1.0 else len(ALL_LOC_TOKENS)-1 for c in box]
        return ''.join([ALL_LOC_TOKENS[i] for i in block_no])
    elif mode == 'text':
        # using text to represent the box
        if space:
            sep = ', '
        else:
            sep = ','
        tmp_format = sep.join(['{' + ':.{}f'.format(prec)+'}']*4)
        a_box = [float(o) for o in box]
        return tmp_format.format(*a_box)
    else:
        raise NotImplementedError

def allbox2str(objects):
    s = []
    for obj in objects:
        s.append('{}: [{}]'.format(obj['class'], box2str(obj['bbox'], 'text', 3, True)))
    return ', '.join(s)

def reshape_box(image, box):
    width, height = image.size
    abs_box = [c*width if i%2==0 else c*height for i,c in enumerate(box)]
    if width == height:
        return box
    elif width > height:
        abs_box[1] += (width - height) // 2
        abs_box[3] += (width - height) // 2
        max_size = width
    else:
        abs_box[0] += (height - width) // 2
        abs_box[2] += (height - width) // 2
        max_size = height
    norm_box = [c/max_size for c in abs_box]
    return norm_box 

def reshape_box_reverse(image, box):
    width, height = image.size
    max_side = max(width, height)
    abs_box = [c*max_side if i%2==0 else c*max_side for i,c in enumerate(box)]
    if width == height:
        return box
    elif width > height:
        abs_box[1] -= (width - height) // 2
        abs_box[3] -= (width - height) // 2
        max_size = width
    else:
        abs_box[0] -= (height - width) // 2
        abs_box[2] -= (height - width) // 2
        max_size = height
    norm_box = [c/width if i%2 ==0 else c/height for i,c in enumerate(abs_box)]
    return norm_box

def resize_image_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    image = image.resize((max_side, max_side))
    return image


def expand2square_fn(pil_img, background_color):
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result