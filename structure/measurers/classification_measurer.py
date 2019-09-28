import torch

from concern import AverageMeter
from concern.config import Configurable


class ClassificationMeasurer(Configurable):
    def __init__(self, **kwargs):
        pass

    def measure(self, batch, output):
        correct = torch.eq(output.cpu(), batch[1]).numpy()
        return correct

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [0]

    def gather_measure(self, raw_metrics, logger=None):
        accuracy_meter = AverageMeter()
        for raw_metric in raw_metrics:
            total = raw_metric.shape[0]
            accuracy = raw_metric.sum() / total
            accuracy_meter.update(accuracy, total)

        return {
            'accuracy': accuracy_meter,
        }
