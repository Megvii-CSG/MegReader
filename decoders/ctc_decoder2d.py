import torch
import torch.nn as nn

from concern.charsets import DefaultCharset


class CTCDecoder2D(nn.Module):
    def __init__(self, in_channels, charset=DefaultCharset(),
                 inner_channels=256, stride=1, blank=0, **kwargs):
        super(CTCDecoder2D, self).__init__()
        self.charset = charset
        from ops import ctc_loss_2d
        self.ctc_loss = ctc_loss_2d

        self.inner_channels = inner_channels
        self.pred_mask = nn.Sequential(
            nn.AvgPool2d(kernel_size=(stride, stride),
                         stride=(stride, stride)),
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.Conv2d(inner_channels, 1, kernel_size=1),
            nn.Softmax(dim=2))

        self.pred_classify = nn.Sequential(
            nn.AvgPool2d(kernel_size=(stride, stride),
                         stride=(stride, stride)),
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.Conv2d(inner_channels, len(charset), kernel_size=1))
        self.blank = blank
        self.tiny = torch.tensor(torch.finfo().tiny, requires_grad=False)
        self.register_buffer('saved_tiny', self.tiny)

    def forward(self, feature, targets=None, lengths=None, train=False,
                masks=None, segs=None):
        tiny = self.saved_tiny
        if isinstance(feature, tuple):
            feature = feature[-1]
        masking = self.pred_mask(feature)
        # mask = masking / torch.max(masking.sum(dim=2, keepdim=True), tiny)
        mask = masking
        classify = self.pred_classify(feature)
        classify = nn.functional.softmax(classify, dim=1)
        if self.training:
            pred = mask * classify  # N, C, H ,W
            pred = torch.log(torch.max(pred, tiny))
            pred = pred.permute(3, 2, 0, 1).contiguous()  # W, H, N, C
            input_lengths = torch.zeros(
                (feature.size()[0], )) + pred.shape[0]
            loss = self.ctc_loss(pred, targets.long(), input_lengths.long().to(
                pred.device), lengths.long()) / lengths.float()
            # return loss, pred.permute(2, 3, 1, 0)
            return loss, pred
        else:
            return classify, mask

    def mask_loss(self, mask, weight, gt):
        batch_size, _, height, _ = mask.shape
        loss = nn.functional.nll_loss(
            (mask.permute(0, 1, 3, 2).reshape(-1, height) + self.saved_tiny).log(),
            gt.reshape(-1), reduction='none').view(batch_size, -1).mean(1)
        return loss * weight

    def classify_loss(self, classify, weight, gt):
        batch_size, classes = classify.shape[:2]
        loss = nn.functional.nll_loss(
            (classify.permute(0, 2, 3, 1).reshape(-1, classes) + self.saved_tiny).log(),
            gt.reshape(-1), reduction='none').view(batch_size, -1)
        position_weights = (gt.view(batch_size, -1) == self.blank).float()
        loss = loss * position_weights

        return loss.mean(1) * weight
