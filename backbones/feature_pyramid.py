import torch.nn as nn


class FeaturePyramid(nn.Module):
    def __init__(self, bottom_up, top_down):
        nn.Module.__init__(self)

        self.bottom_up = bottom_up
        self.top_down = top_down

    def forward(self, feature):
        pyramid_features = self.bottom_up(feature)
        feature = self.top_down(pyramid_features[::-1])
        return feature
