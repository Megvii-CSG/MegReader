import torch
import torch.nn as nn
import numpy as np
import time
import warnings


class CTCLoss2D(nn.Module):
    def __init__(self, blank=0, reduction='mean'):
        r"""The python-implementation of 2D-CTC loss.
        NOTICE: This class is only for the useage of understanding the principle of 2D-CTC.
            Please use `ops.ctc_loss_2d` for practice.
        Args:
            blank (int, optional): blank label. Default :math:`0`.
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the output losses will be divided by the target lengths and
                then the mean over the batch is taken. Default: 'mean'

        Inputs:
            mask: Tensor of size :math:`(T, H, N)` where `H = height`,
                `T = input length`, and `N = batch size`.
                The logarithmized path transition probabilities.
                (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
            classify: Tensor of size :math:`(T, H, N, C)` where `C = number of classes`, `H = height`, `T = input length`, and `N = batch size`.
                The logarithmized character classification probabilities at all possible path pixels.
                (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
            targets: Tensor of size :math:`(N, S)` or `(sum(target_lengths))`.
                Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
            input_lengths: Tuple or tensor of size :math:`(N)`.
                Lengths of the inputs (must each be :math:`\leq T`)
            target_lengths: Tuple or tensor of size  :math:`(N)`.
                Lengths of the targets

        Example::

            >>> ctc_loss = CTCLoss2D()
            >>> N, H, T, C = 16, 8, 32, 20
            >>> mask = torch.randn(T, H, N).log_softmax(1).detach().requires_grad_()
            >>> classify = torch.randn(T, H, N, C).log_softmax(3).detach().requires_grad_()
            >>> targets = torch.randint(1, C, (N, C), dtype=torch.long)
            >>> input_lengths = torch.full((N,), T, dtype=torch.long)
            >>> target_lengths = torch.randint(10, 31, (N,), dtype=torch.long)
            >>> loss = ctc_loss(mask, classify, targets, input_lengths, target_lengths)
            >>> loss.backward()

        Reference:
            2D-CTC for Scene Text Recognition, https://arxiv.org/abs/1907.09705.
        """
        super(CTCLoss2D, self).__init__()
        warnings.warn(
            "NOTICE: This class is only for the useage of understanding the principle of 2D-CTC."
            "Please use `ops.ctc_loss_2d` for practice.")
        self.blank = blank
        self.reduction = reduction
        self.register_buffer('tiny', torch.tensor(torch.finfo().tiny, requires_grad=False))
        self.register_buffer('blank_buffer', torch.tensor([self.blank], dtype=torch.long))
        self.register_buffer('zeros',  torch.log(self.tiny))
        self.registered = False

    def expand_with_blank(self, targets):
        N, S = targets.shape
        blank = self.blank_buffer.repeat(targets.shape)
        expanded_targets = torch.cat([blank.unsqueeze(-1), targets.unsqueeze(-1)], -1)
        expanded_targets = expanded_targets.view(N, -1)
        expanded_targets = torch.cat([expanded_targets, blank[:, 0:1]], dim=-1)
        return expanded_targets

    def log_add(self, a, b):
        x, y = torch.max(a, b), torch.min(a, b)
        return x + torch.log1p(torch.exp(y - x))

    def log_sum(self, x, dim, keepdim=False):
        tiny = self.tiny
        return torch.log(torch.max(
            torch.sum(torch.exp(x), dim=dim, keepdim=keepdim), tiny))

    def safe_log_sum(self, x, keepdim=False):
        result = x[:, 0]
        for i in range(1, x.size(1)):
            result = self.log_add(result, x[:, i])
        if keepdim:
            result = result.unsqueeze(1)
        return result

    def forward(self, mask, classify, targets, input_lengths, target_lengths):
        r"""
        mask: Tensor of size :math:`(T, H, N)` where `H = height`,
            `T = input length`, and `N = batch size`.
            The logarithmized path transition probabilities.
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        classify: Tensor of size :math:`(T, H, N, C)` where `C = number of classes`, `H = height`, `T = input length`, and `N = batch size`.
            The logarithmized character classification probabilities at all possible path pixels.
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`[NOT IMPLEMENTED YET].
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        """
        device = classify.device
        targets = targets.type(torch.long)
        expanded_targets = self.expand_with_blank(targets)
        N, S = expanded_targets.shape
        T, H = classify.shape[:2]
        targets_indices = expanded_targets.repeat(H, 1, 1)

        tiny = self.tiny
        # probability of current time step
        probability = torch.log((torch.zeros(S, H, N) + tiny) / H).to(device)
        probability[0] = classify[0, :, :, self.blank]
        probability[1] = classify[0].gather(
            2, targets_indices[:, :, 1:2]).permute(2, 0, 1)

        # (S - 2, N)
        # The previous token can NOT be ignored when the second previous token is identical with
        # specific token.
        mask_skipping = torch.ne(expanded_targets[:, 2:], expanded_targets[:, :-2]).transpose(0, 1)
        mask_skipping = mask_skipping.unsqueeze(1).type(torch.float).to(device)
        mask_not_skipping = 1 - mask_skipping
        length_indices = torch.linspace(0, S - 1, S).repeat(N, 1, 1).transpose(0, 2).to(device)
        zeros = self.zeros.repeat(S, 1, N).view(S, 1, N)

        count_computable = torch.cat([mask_skipping[0:1] + 1,
                                      mask_skipping[0:1] + 1,
                                      mask_skipping + 1], dim=0)  # (S, N)
        count_computable = torch.cumsum(count_computable, dim=0)

        for timestep in range(1, T):
            mask_uncomputed = (length_indices > count_computable[timestep]).type(torch.float)
            height_summed = self.log_sum(
                probability + mask[timestep - 1].unsqueeze(0), dim=1, keepdim=True)

            height_summed = height_summed * (1 - mask_uncomputed) + zeros * mask_uncomputed

            new_probability1 = self.log_add(height_summed[1:], height_summed[:-1])  # (S-1, H, N)
            new_probability2 = self.log_add(new_probability1[1:], height_summed[:-2]) * mask_skipping\
                + new_probability1[1:] * mask_not_skipping

            new_probability = torch.cat([height_summed[:1],
                                         new_probability1[:1],
                                         new_probability2], dim=0)
            probability = new_probability + \
                classify[timestep].gather(2, targets_indices).permute(2, 0, 1)
        probability = self.safe_log_sum(probability + mask[T-1].unsqueeze(0))  # (S, N)
        lengths = (target_lengths * 2 + 1).unsqueeze(0)
        loss = self.log_add(probability.gather(0, lengths - 1), probability.gather(0, lengths - 2))
        loss = loss.squeeze(0)
        if self.reduction == 'mean':
            return -(loss / target_lengths.type(torch.float))
        elif self.reduction == 'sum':
            return -loss.sum()
        return -loss
