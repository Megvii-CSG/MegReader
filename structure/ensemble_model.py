import glob
import os
import pickle

import numpy as np

import torch
import torch.nn as nn

from backbones import *
from decoders import *


class EnsembleModel(nn.Module):
    def __init__(self, models, *args, **kwargs):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleDict(models)

    def forward(self, batch, select_key=None, training=False):
        pred = dict()

        for key, module in self.models.items():
            if select_key is not None and key != select_key:
                continue
            pred[key] = module(batch, training)
        return pred
