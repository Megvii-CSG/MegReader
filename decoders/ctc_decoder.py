#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ctc_decoder.py
# Author            : Zhaoyi Wan <wanzhaoyi@megvii.com>
# Date              : 18.12.2018
# Last Modified Date: 20.01.2019
# Last Modified By  : Zhaoyi Wan <wanzhaoyi@megvii.com>

import torch
import torch.nn as nn
from concern.charsets import DefaultCharset


class CTCDecoder(nn.Module):
    def __init__(self, in_channels, charset=DefaultCharset(), inner_channels=256, **kwargs):
        super(CTCDecoder, self).__init__()
        self.ctc_loss = nn.CTCLoss(reduction='mean')
        self.inner_channels = inner_channels
        self.encode = self._init_encoder(in_channels)

        self.pred_conv = nn.Conv2d(
            inner_channels, len(charset), kernel_size=1, bias=True, padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

        self.blank = 0
        if 'blank' in kwargs:
            self.blank = kwargs['blank']

    def _init_encoder(self, in_channels, stride=(2, 1), padding=(0, 1)):
        encode = nn.Sequential(
            self.conv_bn_relu(in_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels,
                              kernel_size=(2, 3),
                              stride=stride, padding=padding),
        )
        return encode

    def conv_bn_relu(self, input_channels, output_channels,
                     kernel_size=3, stride=1, padding=1):
        return nn.Sequential(nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),)

    def forward(self, feature, targets=None, lengths=None, train=False):
        pred = self.encode(feature)
        pred = self.pred_conv(pred)
        if train:
            pred = self.softmax(pred)
            pred = pred.select(2, 0)  # N, C, W
            pred = pred.permute(2, 0, 1)  # W, N, C
            input_lengths = torch.zeros((feature.size()[0], ), dtype=torch.int) + 32
            loss = self.ctc_loss(pred, targets, input_lengths, lengths)
            return loss, pred.permute(1, 2, 0)
        else:
            pred = nn.functional.softmax(pred, dim=1)
            return pred
