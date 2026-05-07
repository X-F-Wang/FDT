import argparse
import pathlib
import copy
import os
import cv2
import torch.nn.functional as F
from PIL import Image
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
from torchvision import transforms

def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(transforms.ToTensor()(Image.open(img_path).convert('RGB')))
    )


def save_img(img, img_path):
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    plt.savefig(img_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sr')
    parser.add_argument('-lr')
    parser.add_argument('-resolution')
    parser.add_argument('-output', default='output.png')
    args = parser.parse_args()

    img_path = args.sr
    lr_path = args.lr
    sr_img = read_img(img_path).cuda()
    lr_img = read_img(lr_path).cuda()

    h, w = list(map(int, args.resolution.split(',')))

    coords = []

    coord = utils.make_coord((h, w), flatten=False).unsqueeze(0)

    coords.append(coord)
    # print(coord.shape)
    sr_img = sr_img.unsqueeze(0)
    lr_img = lr_img.unsqueeze(0)

    lr_img = F.grid_sample(lr_img, coords[0].flip(-1).cuda(), mode='bilinear', align_corners=False)
    # sr_img = F.grid_sample(sr_img, coords[0].flip(-1).cuda(), mode='bilinear', align_corners=False)
    # transforms.ToPILImage()(lr_img.squeeze().cpu()
    #                         ).save('lr_img.png')
    cha = abs(sr_img - lr_img).squeeze().cpu()
    total = torch.zeros((h, w))
    for i in range(3):
        total += cha[i, :, :]

    total = total / 3
    print(total.shape)

    total = {'total': total}
    data = total['total']
    data = data[:, :]

    cmap = colors.ListedColormap(['blue'])
    # 可视化均方误差地图
    plt.imshow(data, cmap='YlGnBu_r')
    plt.colorbar()
    # 设置图像标题和坐标轴标签
    plt.title('Mean Error Map')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图像
    plt.show()
    plt.savefig(args.output)

