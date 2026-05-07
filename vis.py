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
import utils
import copy
def build_transforms(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transforms4display(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

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

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(transforms.ToTensor()(Image.open(img_path).convert('RGB')))
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model')
    parser.add_argument('-name')
    parser.add_argument('-scale',default=2)
    args = parser.parse_args()

    # %%
    '''
    build model
    '''
    img_size = 224

    # SMT
    # model = SMT(
    #     embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[4, 4, 4, 2],
    #     qkv_bias=True, depths=[3, 4, 18, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, ).cuda()
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    # model.eval()
    # %%
    '''
    build data transform
    '''
    eval_transforms = build_transforms(img_size, center_crop=False)
    display_transforms = build_transforms4display(img_size, center_crop=False)
    # %%
    '''
    load checkpoint
    '''
    # visualize modulator
    upsampler = nn.Upsample(scale_factor=4, mode='bilinear')

    # img_folder = "/home/jinchen/TEST/clit/assets/"
    img_paths = "/home/SCISR/SCI1K-Test/X2/LR/00065x2.png"
    #img_paths = "/home/SCISR/SCI1K-Test/X4/HR/00111.png"
    # for i, img_path in enumerate(img_paths):
    # img = Image.open(img_folder + img_path)
    # img_t = eval_transforms(img)
    # img_d = display_transforms(img)
    img_d = read_img(img_paths)
    hr_h, hr_w = img_d.shape[-2:]
    scale = args.scale
    lr_img = resize_fn(img_d, (round(hr_h / scale), round(hr_w / scale)))

    inp = lr_img.unsqueeze(0).cuda()

    inp_h, inp_w = inp.shape[-2:]

    scale_list = system_scale(scale)

    coords = []

    for idx in range(len(scale_list)):
        h = round(inp_h * np.prod(scale_list[:idx + 1]))
        w = round(inp_w * np.prod(scale_list[:idx + 1]))

        coord = utils.make_coord((h, w), flatten=False).unsqueeze(0)

        coords.append(coord)

    cell = torch.ones(2).unsqueeze(0)
    cell[:, 0] *= 2. / hr_h
    cell[:, 1] *= 2. / hr_w
    for idx in range(len(coords)):
        coords[idx] = coords[idx].cuda()

    cell = cell.cuda()


    data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1)

    # out = model(img_t.unsqueeze(0).cuda())


    with torch.no_grad():
        preds = preds = model.chop_forward((inp - inp_sub) / inp_div, coords, cell, 30000)


    fig = plt.figure(figsize=(36, 8))

    # ori image
    fig.add_subplot(1, 2, 1)
    img2d = img_d.permute(1, 2, 0).cpu().detach().contiguous().numpy()
    x = plt.imshow(img_d.permute(1, 2, 0).cpu().detach().contiguous().numpy())
    plt.axis('off')
    x.axes.get_xaxis().set_visible(False)
    x.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=None, hspace=None)

    # Modulator vis in stage 1
    fig.add_subplot(1, 2, 2)
    # print(model)
    modulator = torch.abs((model.feat)).mean(1, keepdim=True)
    # print(modulator.size())
    modulator = upsampler(modulator)
    x = plt.imshow((modulator.squeeze(1)).permute(1, 2, 0).cpu().detach().contiguous().numpy())
    plt.axis('off')
    x.axes.get_xaxis().set_visible(False)
    x.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Modulator vis in stage 2
    #fig.add_subplot(1, 3, 3)
    #modulator2 = torch.abs((model.feat)).mean(1, keepdim=True)
    # print(modulator.size())
    #modulator = upsampler(modulator2)
    #x = plt.imshow((modulator.squeeze(1)).permute(1, 2, 0).cpu().detach().contiguous().numpy())
    #plt.axis('off')
    #x.axes.get_xaxis().set_visible(False)
    #x.axes.get_yaxis().set_visible(False)
    #plt.subplots_adjust(wspace=0, hspace=0)
    #
    # # Modulator vis in stage 3
    # fig.add_subplot(1, 4, 4)
    # modulator = torch.abs((model.modulator3)).mean(1, keepdim=True)
    # # print(modulator.size())
    # modulator = upsampler(modulator)
    # x = plt.imshow((modulator.squeeze(1)).permute(1, 2, 0).cpu().detach().contiguous().numpy())
    # plt.axis('off')
    # x.axes.get_xaxis().set_visible(False)
    # x.axes.get_yaxis().set_visible(False)

    plt.savefig('./{}.png'.format(args.name), dpi=600)


