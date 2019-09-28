import torch
import torch.nn as nn
import config
from .ctc_loss import CTCLoss
from concern.charsets import DefaultCharset


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNNDecoder(nn.Module):

    def __init__(self, charset=DefaultCharset(),
                 inner_channels=256, in_channels=256, need_reduce=False, reduce_func=None, loss_func='pytorch'):

        super().__init__()
        rnn_input = in_channels
        if need_reduce:
            rnn_input = inner_channels
        self.rnn = nn.Sequential(
            BidirectionalLSTM(rnn_input, inner_channels, inner_channels),
            BidirectionalLSTM(inner_channels, inner_channels, len(charset))
        )
        self.inner_channels = inner_channels
        if need_reduce:
            if reduce_func == 'conv':
                self.fpn2rnn = self._init_conv(in_channels)
            elif need_reduce and reduce_func == 'pooling':
                self.fpn2rnn = self._init_pooling()
        self.softmax = nn.Softmax()
        if loss_func == 'pytorch':
            self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        else:
            self.ctc_loss = CTCLoss()

    def _init_conv(self, in_channels, stride=(2, 1), padding=(0, 1)):
        encode = nn.Sequential(
            self.conv_bn_relu(in_channels, self.inner_channels),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),

            # nn.AdaptiveMaxPool2d((1, None))
            # self.conv_bn_relu(self.inner_channels, self.inner_channels,
            #                    kernel_size=(2, 3),
            #                    stride=stride, padding=padding),
        )
        return encode

    def _init_pooling(self):
        return nn.AdaptiveMaxPool2d((1, None))

    def conv_bn_relu(self, input_channels, output_channels,
                     kernel_size=3, stride=1, padding=1):
        return nn.Sequential(nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),)


    def forward(self, feature, targets=None, lengths=None, train=False):
        b, c, h, w = feature.size()
        # print(feature)
        # print(feature.size())
        if h > 1:
            feature = self.fpn2rnn(feature)
            b, c, h, w = feature.size()
        assert h == 1, "the height of conv must be 1"

        feature = feature.squeeze(2)  # N, C, W
        feature = feature.permute(2, 0, 1)  # W, N, C
        for r in self.rnn:
            r.rnn.flatten_parameters()
        pred = self.rnn(feature)

        if train:
            pred = nn.functional.log_softmax(pred, dim=2).to(torch.float64)
            pred_size = torch.Tensor([pred.size(0)] * b).int()
            loss = self.ctc_loss(pred, targets, pred_size, lengths)
            return loss, pred
        else:
            pred = pred.permute(1, 2, 0)
            pred = pred.unsqueeze(2)
            pred = nn.functional.softmax(pred, dim=1)
            return pred
