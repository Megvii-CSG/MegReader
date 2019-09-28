import torch
import torch.nn as nn
import torch.nn.functional as F


class EASTDecoder(nn.Module):
    def __init__(self, channels=256, heatmap_ratio=1.0, densebox_ratio=0.01, densebox_rescale_factor=512):
        nn.Module.__init__(self)

        self.heatmap_ratio = heatmap_ratio
        self.densebox_ratio = densebox_ratio
        self.densebox_rescale_factor = densebox_rescale_factor

        self.head_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2, padding=0),
        )

        self.heatmap_pred_layer = nn.Sequential(
            nn.Conv2d(channels // 4, 1, kernel_size=1, stride=1, padding=0),
        )

        self.densebox_pred_layer = nn.Sequential(
            nn.Conv2d(channels // 4, 8, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input, label, meta, train):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        densebox = label['densebox']
        densebox_weight = label['densebox_weight']

        feature = self.head_layer(input)
        heatmap_pred = self.heatmap_pred_layer(feature)
        densebox_pred = self.densebox_pred_layer(feature) * self.densebox_rescale_factor

        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        densebox_loss = F.mse_loss(densebox_pred, densebox, reduction='none')
        densebox_loss = (densebox_loss * densebox_weight).mean(dim=(1, 2, 3))

        loss = heatmap_loss * self.heatmap_ratio + densebox_loss * self.densebox_ratio

        pred = {
            'heatmap': F.sigmoid(heatmap_pred),
            'densebox': densebox_pred,
        }
        metrics = {
            'heatmap_loss': heatmap_loss,
            'densebox_loss': densebox_loss,
        }
        if train:
            return loss, pred, metrics
        else:
            return pred
