from distutils.command import install
from constants import ALL_LOC_TOKENS
from collections import defaultdict
import os, re
from PIL import ImageDraw, ImageFont
from torchvision.ops import box_iou
import torch
import numpy as np

color_map = {
    'red': '#FF0000',
    'blue': '#0000FF',
    'green': '#00FF00',
    'yellow': '#FFFF00',
    'pink': '#FFC0CB',
    'purple': '#A020F0',
    'orange': '#FFA500',
    'grey': '#BEBEBE',
    'brown': '#A52A2A',
    'black': '#000000'
}

colors = ['red','blue','green','yellow','pink','purple','orange','grey','brown','black']

def extract_box(output):
    loc2id = {loc:i for i,loc in enumerate(ALL_LOC_TOKENS)}
    tmp_box = [loc2id[k] for k in output.split() if k in loc2id]
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    tmp_box = [1/(2*len(ALL_LOC_TOKENS)) + i/len(ALL_LOC_TOKENS) for i in tmp_box]
    return tmp_box

def extract_box_str(output, mistral=False):
    if mistral:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+</coor>'
    else:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+ </coor>'
    boxes = re.findall(pattern, output)
    tmp_box = []
    for b in boxes:
        if mistral:
            tmp_box.extend([float(k) for k in b[7:-7].split(',')])
        else:
            tmp_box.extend([float(k) for k in b[7:-8].split(',')])
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def extract_all_box_str(output, mistral=False):
    if mistral:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+</coor>'
    else:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+ </coor>'
    boxes = re.findall(pattern, output)
    tmp_all_box = []
    for b in boxes:
        if mistral:
            tmp_box = [float(k) for k in b[7:-7].split(',')]
        else:
            tmp_box = [float(k) for k in b[7:-8].split(',')]
        # tmp_box = [float(k) for k in b[7:-8].split(',')]
        if len(tmp_box) > 4:
            tmp_box = tmp_box[:4]
        elif len(tmp_box) != 4:
            tmp_box = None
        tmp_all_box.append(tmp_box)
    return tmp_all_box

def remove_all_box_str(output, mistral=False):
    if mistral:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+</coor>'
    else:
        pattern = '<coor> [0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+ </coor>'
    boxes = re.findall(pattern, output)
    tmp_all_box = []
    for b in boxes:
        output = output.replace(b, '')
    return output

def extract_box_str_space(output):
    pattern = '<coor> [0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+ </coor>'
    boxes = re.findall(pattern, output)
    tmp_box = []
    for b in boxes:
        tmp_box.extend([float(k) for k in b[7:-8].split(',')])
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def extract_box_str_llava(output):
    pattern = '\[[0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+\]'
    boxes = re.findall(pattern, output)
    tmp_box = []
    for b in boxes:
        tmp_box.extend([float(k) for k in b[1:-1].split(',')])
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def extract_box_str_llava16(output, mistral=False):
    if mistral:
        pattern = '\([0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+\)'
    else:
        pattern = '\[[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+,[0-1]\.[0-9]+\]'
    boxes = re.findall(pattern, output)
    tmp_box = []
    for b in boxes:
        tmp_box.extend([float(k) for k in b[1:-1].split(',')])
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def extract_box_str_qwenvl(output):
    pattern = "<box>\S*</box>"
    boxes = re.findall(pattern, output)
    if len(boxes) == 0:
        return None
    tmp_box = []
    for b in boxes:
        tmp_b = b.replace('<box>', '').replace('</box>', '').replace('(', '').replace(')','')
        tmp_box.extend([int(k)/1000 for k in tmp_b.split(',')])
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def extract_box_str_minigptv2(output):
    pattern = "\d+"
    ints = [int(i) for i in re.findall(pattern, output)]
    if len(ints) < 4:
        return None
    tmp_box = []
    for b in ints:
        tmp_box.append(b/100)
    if len(tmp_box) > 4:
        tmp_box = tmp_box[:4]
    elif len(tmp_box) != 4:
        return None
    assert(len(tmp_box) == 4)
    return tmp_box

def draw_box(image, gt_box, pred_box=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    gt_box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(gt_box)]
    if pred_box is not None:
        pred_box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(pred_box)]
        draw.rectangle(pred_box_size, outline=(255,0,0), width=2)
    draw.rectangle(gt_box_size, outline=(0,255,0), width=2)
    del(draw)
    return image

def draw_all_box(image, boxes=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()# ImageFont.truetype(font='simhei.tff', size=np.floor(1.5e-2 * np.shape(image)[1] + 10).astype('int32'))
    for obj in boxes:
        box = [float(b) for b in obj['bbox']]
        box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(box)]
        tmp_color = tuple(np.random.randint(0, 255, size=[3]))
        draw.rectangle(tuple(box_size), outline=tmp_color, width=2)
        label_ = obj['class']
        label_size = draw.textsize(label_, font)
        text_origin = np.array([box[0], box[1] + 0.2 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tmp_color)
        # draw.text(text_origin, str(label_), fill=(255, 255, 255), font=font)
    del(draw)
    return image

def draw_all_box(image, boxes=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()# ImageFont.truetype(font='simhei.tff', size=np.floor(1.5e-2 * np.shape(image)[1] + 10).astype('int32'))
    for obj in boxes:
        box = [float(b) for b in obj['bbox']]
        box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(box)]
        tmp_color = tuple(np.random.randint(0, 255, size=[3]))
        draw.rectangle(tuple(box_size), outline=tmp_color, width=2)
        label_ = obj['class']
        label_size = draw.textsize(label_, font)
        text_origin = np.array([box[0], box[1] + 0.2 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tmp_color)
        # draw.text(text_origin, str(label_), fill=(255, 255, 255), font=font)
    del(draw)
    return image

def draw_all_box_colored(image, boxes=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    short_side = min(width, height)
    font = ImageFont.load_default()# ImageFont.truetype(font='simhei.tff', size=np.floor(1.5e-2 * np.shape(image)[1] + 10).astype('int32'))
    for i, obj in enumerate(boxes):
        box = [float(b) for b in obj['bbox']]
        box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(box)]
        tmp_color = color_map[colors[i]]
        draw.rectangle(tuple(box_size), outline=tmp_color, width=int(0.005*short_side))
        label_ = obj['class']
        label_size = draw.textsize(label_, font)
        text_origin = np.array([box[0], box[1] + 0.2 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tmp_color)
        # draw.text(text_origin, str(label_), fill=(255, 255, 255), font=font)
    del(draw)
    return image

def draw_all_box_raw(image, boxes=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    short_side = min(width, height)
    # font = ImageFont.load_default()# ImageFont.truetype(font='simhei.tff', size=np.floor(1.5e-2 * np.shape(image)[1] + 10).astype('int32'))
    for i, obj in enumerate(boxes):
        box = [float(b) for b in obj]
        box_size = [c * width if (i % 2==0) else c * height for i,c in enumerate(box)]
        tmp_color = color_map[colors[i% len(color_map)]]
        draw.rectangle(tuple(box_size), outline=tmp_color, width=int(0.005*short_side))
        # label_ = obj['class']
        # label_size = draw.textsize(label_, font)
        # text_origin = np.array([box[0], box[1] + 0.2 * label_size[1]])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tmp_color)
        # draw.text(text_origin, str(label_), fill=(255, 255, 255), font=font)
    del(draw)
    return image

def res2mme(res, dataset, output_dir):
    split2res = {}
    for item in res:
        index = item['item_id'].split('_')[1]
        meta = dataset.meta[int(index)]
        split = '_'.join(meta['id'].split('_')[:-2])
        if split in split2res:
            current_dict = split2res[split]
        else:
            current_dict = defaultdict(list)
            split2res[split] = current_dict
        current_dict[meta['image']].append('\t'.join([meta['image'], meta['question'], meta['answer'], item['prediction']]))
    for k,res in split2res.items():
        with open(os.path.join(output_dir, '{}.txt'.format(k)), 'w') as wf:
            for img, ques in res.items():
                assert len(ques)==2, ques
                wf.write(ques[0]+'\n')
                wf.write(ques[1]+'\n')
        

def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    if box1 is None or box2 is None:
        return 0
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

def cal_iou_pt(box1, box2):
    if box1 is None or box2 is None:
        return 0
    box1 = torch.tensor(box1).unsqueeze(0)
    box2 = torch.tensor(box2).unsqueeze(0)
    return box_iou(box1*1000, box2*1000)[0,0].item()

def cal_iou_pt(box1, box2):
    if box1 is None or box2 is None:
        return 0
    box1 = torch.tensor(box1).unsqueeze(0)
    box2 = torch.tensor(box2).unsqueeze(0)
    return box_iou(box1*1000, box2*1000)[0,0].item()

def cal_nn_iou(boxes):
    # input: boxes: n*4 [[xmin, ymin, xmax, ymax]]
    # output: ious: n*n
    pt_boxes = torch.tensor(boxes)
    return box_iou(pt_boxes, pt_boxes)

def check_iou(boxes, threshold=0.3):
    iou_mat = cal_nn_iou(boxes)
    n = iou_mat.shape[0]
    iou_mat[torch.arange(n), torch.arange(n)] = 0
    return iou_mat.max().item() > threshold
