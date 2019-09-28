import torch
import torch.nn as nn
import torch.nn.functional as F


class TextsnakeDecoder(nn.Module):

    def __init__(self, channels=256):
        nn.Module.__init__(self)

        self.head_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2, padding=0),
        )

        self.pred_layer = nn.Sequential(
            nn.Conv2d(channels // 4, 7, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def ohem(predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()

        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))

        loss_pos = F.cross_entropy(predict, target, reduction='none')[pos].sum()

        loss_neg = F.cross_entropy(predict, target, reduction='none')[neg]
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, label, meta, train):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param label: (dict)
            :param tr_mask: (Variable), TR target, (BS, H, W)
            :param tcl_mask: (Variable), TCL target, (BS, H, W)
            :param sin_map: (Variable), sin target, (BS, H, W)
            :param cos_map: (Variable), cos target, (BS, H, W)
            :param radius_map: (Variable), radius target, (BS, H, W)
            :param train_mask: (Variable), training mask, (BS, H, W)
            :return: loss_tr, loss_tcl, loss_radius, loss_sin, loss_cos
        """
        tr_mask = label['tr_mask']
        tcl_mask = label['tcl_mask']
        sin_map = label['sin_map']
        cos_map = label['cos_map']
        radius_map = label['radius_map']
        train_mask = label['train_mask']

        feature = self.head_layer(input)
        pred = self.pred_layer(feature)

        tr_out = pred[:, :2]
        tcl_out = pred[:, 2:4]
        sin_out = pred[:, 4]
        cos_out = pred[:, 5]
        radius_out = pred[:, 6]

        tr_pred = tr_out.permute(0, 2, 3, 1).reshape(-1, 2)  # (BSxHxW, 2)
        tcl_pred = tcl_out.permute(0, 2, 3, 1).reshape(-1, 2)  # (BSxHxW, 2)
        sin_pred = sin_out.reshape(-1)  # (BSxHxW,)
        cos_pred = cos_out.reshape(-1)  # (BSxHxW,)
        radius_pred = radius_out.reshape(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        train_mask = train_mask.view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.reshape(-1)
        tcl_mask = tcl_mask.reshape(-1)
        radius_map = radius_map.reshape(-1)
        sin_map = sin_map.reshape(-1)
        cos_map = cos_map.reshape(-1)

        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        loss_tcl = F.cross_entropy(tcl_pred, tcl_mask.long(), reduction='none')[train_mask * tr_mask].mean()

        # geometry losses
        ones = radius_map.new(radius_pred[tcl_mask].size()).fill_(1.).float()
        loss_radius = F.smooth_l1_loss(radius_pred[tcl_mask] / radius_map[tcl_mask], ones)
        loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
        loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        loss = loss_tr + loss_tcl + loss_radius + loss_sin + loss_cos
        pred = {
            'tr_pred': F.softmax(tr_out, dim=1)[:, 1],
            'tcl_pred': F.softmax(tcl_out, dim=1)[:, 1],
            'sin_pred': sin_out,
            'cos_pred': cos_out,
            'radius_pred': radius_out,
        }
        metrics = {
            'loss_tr': loss_tr,
            'loss_tcl': loss_tcl,
            'loss_radius': loss_radius,
            'loss_sin': loss_sin,
            'loss_cos': loss_cos,
        }
        if train:
            return loss, pred, metrics
        else:
            return pred
