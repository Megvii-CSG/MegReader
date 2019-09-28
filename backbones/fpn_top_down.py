import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNTopDown(nn.Module):
    def __init__(self, pyramid_channels, feature_channel):
        nn.Module.__init__(self)

        self.reduction_layers = nn.ModuleList()
        for pyramid_channel in pyramid_channels:
            reduction_layer = nn.Conv2d(pyramid_channel, feature_channel, kernel_size=1, stride=1, padding=0, bias=False)
            self.reduction_layers.append(reduction_layer)

        self.merge_layer = nn.Conv2d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pyramid_features):
        feature = None
        for pyramid_feature, reduction_layer in zip(pyramid_features, self.reduction_layers):
            pyramid_feature = reduction_layer(pyramid_feature)
            if feature is None:
                feature = pyramid_feature
            else:
                feature = self.upsample_add(feature, pyramid_feature)
        feature = self.merge_layer(feature)
        return feature
