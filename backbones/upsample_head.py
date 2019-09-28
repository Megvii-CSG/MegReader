import torch.nn as nn


def SimpleUpsampleHead(feature_channel, layer_channels):
    modules = []
    modules.append(nn.Conv2d(feature_channel, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False))
    for layer_index in range(len(layer_channels) - 1):
        modules.extend([
            nn.BatchNorm2d(layer_channels[layer_index]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layer_channels[layer_index], layer_channels[layer_index + 1], kernel_size=2, stride=2, padding=0, bias=False),
        ])
    return nn.Sequential(*modules)
