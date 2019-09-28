import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.kernels = [3, 3, 3, 3, 3, 3, 2]
        self.paddings = [1, 1, 1, 1, 1, 1, 0]
        self.strides = [1, 1, 1, 1, 1, 1, 1]
        self.channels = [64, 128, 256, 256, 512, 512, 512, nc]

        conv0 = nn.Sequential(
            self._make_layer(0),
            nn.MaxPool2d((2, 2))
        )
        conv1 = nn.Sequential(
            self._make_layer(1),
            nn.MaxPool2d((2, 2))
        )
        conv2 = self._make_layer(2, True)
        conv3 = nn.Sequential(
            self._make_layer(3),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )
        conv4 = self._make_layer(4, True)
        conv5 = nn.Sequential(
            self._make_layer(5),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )
        conv6 = self._make_layer(6, True)

        self.cnn = nn.Sequential(
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6
        )


    def _make_layer(self, i, batch_normalization=False):
        in_channel = self.channels[i - 1]
        out_channel = self.channels[i]
        layer = list()
        layer.append(nn.Conv2d(in_channel, out_channel, self.kernels[i], self.strides[i], self.paddings[i]))
        if batch_normalization:
            layer.append(nn.BatchNorm2d(out_channel))
        else:
            layer.append(nn.ReLU())
        return nn.Sequential(*layer)

    def forward(self, input):
        # conv features
        return self.cnn(input)


def crnn_backbone(imgH=32, nc=3, nclass=37, nh=256):
    return CRNN(imgH, nc, nclass, nh)
