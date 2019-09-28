import torch

from concern import AverageMeter
from concern.config import Configurable


class GridSamplingMeasurer(Configurable):
    def __init__(self, **kwargs):
        pass

    def measure(self, batch, output):
        return 0

    def validate_measure(self, batch, output):
        return 1, [0]

    def gather_measure(self, raw_metrics, logger=None):
        return {
            'accuracy': 0,
        }
