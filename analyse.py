import logging
from os import path as osp

import yaml
import argparse
import torch
from utils import set_save_path

from basicsr.utils import get_root_logger, get_time_str
from analy import  get_model_flops, get_model_activation
import os
import models

def main(config_, savepath, img_size: tuple = (3, 128, 256), coord: tuple = (128, 128, 2),\
         cell: tuple = (2,)):  # noqa
    # parse options, set distributed setting, set random seed
    global config, log, writer
    config = config_
    log, writer = set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model = models.make(config['model']).cuda()

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False  # noqa
    log_file = osp.join(savepath, f"analyse__{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    # create model

    logger.info(f'Analyzing {model.__class__.__name__}...')

    # analyse Params
    logger.info(f"#Params [M]: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # analyse FLOPs
    flops = get_model_flops(model, img_size, coord, cell, False)
    logger.info(f"#basicsr:FLOPs [G]: {flops / 10 ** 9}")

    # analyse Acts and Conv
    acts, conv = get_model_activation(model, img_size, coord, cell)
    logger.info(f"#Acts [M]: {acts / 10 ** 6}")
    logger.info(f"#Conv: {conv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',default='./configs/train_edsr-sronet.yaml')
    parser.add_argument('-name', default=None)
    parser.add_argument('-tag', default=None)
    parser.add_argument('-gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)


    main(config, save_path)
