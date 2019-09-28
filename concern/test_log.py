#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_log.py
# Author            : Zhaoyi Wan <wanzhaoyi@megvii.com>
# Date              : 06.01.2019
# Last Modified Date: 06.01.2019
# Last Modified By  : Zhaoyi Wan <wanzhaoyi@megvii.com>

import argparse
import os
import torch
import torch.utils.data as data

from models import Builder
from concern import Log, AverageMeter
from data import NoriDataset
from tqdm import tqdm
import time
import config
import glob

parser = argparse.ArgumentParser(description='Text Recognition Training')
parser.add_argument('--name', default='', type=str)
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=1000000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--start_epoch', default=0, type=int, help='Begin counting epoch starting from this value (should be used with resume)')
parser.add_argument('--use_isilon', default=True, type=bool, help='Use isilon to save logs and checkpoints')
parser.add_argument('--num_gpus', default=4, type=int, help='number of gpus to use')
parser.add_argument('--backbone', default='resnet50_fpn', type=str, help='The backbone want to use')
parser.add_argument('--decoder', default='1d_ctc', type=str, help='The ctc formulation want to use')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--aug', default=True, type=bool, help='add data augmentation')
parser.add_argument('--verbose', default=True, type=bool, help='show verbose info')
parser.add_argument('--validate', action='store_true', dest='validate', help='Validate during training')
parser.add_argument('--no-validate', action='store_false', dest='validate', help='Validate during training')
parser.set_defaults(validate=True)
parser.add_argument('--nori', default='/unsullied/sharefs/_csg_algorithm/OCR/zhangjian02/data/ocr-data/synth-data/SynthText/croped_sorted.nori', type=str, help='nori_file_path')
parser.add_argument('--test_nori', default='/unsullied/sharefs/wanzhaoyi/data/text-recognition-test/*.nori', type=str, help='nori_file_path for validation')

args = parser.parse_args()
name = args.name
if name == '':
    name = args.backbone + '-' + args.decoder
logger = Log('test_logger', args.use_isilon, args.verbose)
logger.args(args)


epoch = 0
for e in range(3):
    for i in range(28385):
        loss = i * 1e-7
        logger.add_scalar('loss', loss, i + epoch * 28385);

        if i % 100 == 0:
                logger.info('iter: %6d, epoch: %3d, loss: %.6f, lr: %f' % (i, epoch, loss, 1e-4))

    epoch += 1

