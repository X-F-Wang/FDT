# --------------------------------------------------------------------------------
# Analyze super-resolution models using LAM.
# Official GitHub: https://github.com/X-Lowlevel-Vision/LAM_Demo
#
# Modified by Jinpeng Shi (https://github.com/jinpeng-s)
# --------------------------------------------------------------------------------
# import logging
# import os.path
# from os import path as osp
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
import argparse
import os
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
import models
import copy
import utils
# from basicsr.models import build_model
# from basicsr.utils import get_env_info, get_root_logger, get_time_str
# from basicsr.utils.options import dict2str

# import archs  # noqa
# import data  # noqa
# import models  # noqa
from lam_utils import parse_options, make_exp_dirs, get_model_interpretation
import argparse

def interpret_pipeline(model, img_path, name):  # noqa
        img, di = get_model_interpretation(model, img_path, 110, 150,
                                           use_cuda=True )
        os.makedirs("./lam", exist_ok=True)
        img.save(f"./lam/{name}.png")

def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(transforms.ToTensor()(Image.open(img_path).convert('RGB')))
    )
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )
def system_scale(scale, scale_base=4):
    scale_list = []
    s = copy.copy(scale)

    if s <= scale_base:
        scale_list.append(s)
    else:
        scale_list.append(scale_base)
        s = s / scale_base
        while s > 1:
            if s >= scale_base:
                scale_list.append(scale_base)
            else:
                scale_list.append(s)
            s = s / scale_base

    return scale_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model')
    parser.add_argument('-name')
    parser.add_argument('-scale',default=4)
    parser.add_argument('-input')
    parser.add_argument('-gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    img_paths = args.input

    interpret_pipeline(model, img_paths,name=args.name)