import torch
import torch.nn as nn


class ClassificationDecoder(nn.Module):

    def __init__(self):
        super(ClassificationDecoder, self).__init__()

        self.fc = torch.nn.Linear(256, 10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, feature_map, targets=None, train=False):
        x = torch.max(torch.max(feature_map, dim=3)[0], dim=2)[0]
        x = self.fc(x)
        pred = x
        if train:
            loss = self.criterion(pred, targets)
            return loss, pred
        else:
            return pred
