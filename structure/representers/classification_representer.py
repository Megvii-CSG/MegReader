import torch
from concern.config import Configurable


class ClassificationRepresenter(Configurable):
    def __init__(self, **kwargs):
        pass

    def represent(self, batch, pred):
        return torch.argmax(pred, dim=1)
