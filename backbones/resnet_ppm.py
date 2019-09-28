import torch
import torch.nn as nn

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_dilated import ResnetDilated
from .ppm import PPMDeepsup


def resnet50dilated_ppm(resnet_pretrained=False, **kwargs):
    resnet = resnet50(pretrained=resnet_pretrained)
    resnet_dilated = ResnetDilated(resnet, dilate_scale=8)
    ppm = PPMDeepsup(**kwargs)
    return nn.Sequential(resnet_dilated, ppm)
