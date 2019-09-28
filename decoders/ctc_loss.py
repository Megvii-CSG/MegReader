import torch
import torch.nn as nn
import numpy as np
import time

class CTCLoss(nn.Module):

    def __init__(self, blank=0, reduction='mean'):
        r"""The Connectionist Temporal Classification loss.

	Args:
	    blank (int, optional): blank label. Default :math:`0`.
	    reduction (string, optional): Specifies the reduction to apply to the output:
		'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
		'mean': the output losses will be divided by the target lengths and
		then the mean over the batch is taken. Default: 'mean'

	Inputs:
	    log_probs: Tensor of size :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
		`T = input length`, and `N = batch size`.
		The logarithmized probabilities of the outputs
		(e.g. obtained with :func:`torch.nn.functional.log_softmax`).
	    targets: Tensor of size :math:`(N, S)` or `(sum(target_lengths))`.
		Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
	    input_lengths: Tuple or tensor of size :math:`(N)`.
		Lengths of the inputs (must each be :math:`\leq T`)
	    target_lengths: Tuple or tensor of size  :math:`(N)`.
		Lengths of the targets


	Example::

	    >>> ctc_loss = CTCLoss()
	    >>> log_probs = torch.randn(12, 16, 20).log_softmax(2).detach().requires_grad_()
	    >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
	    >>> input_lengths = torch.full((16,), 12, dtype=torch.long)
	    >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
	    >>> loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
	    >>> loss.backward()

	Reference:
	    A. Graves et al.: Connectionist Temporal Classification:
	    Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
	    https://www.cs.toronto.edu/~graves/icml_2006.pdf
	"""
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction


    def expand_with_blank(self, targets):
        N, S = targets.shape
        blank = torch.tensor([self.blank], dtype=torch.long).repeat(targets.shape)
        expanded_targets = torch.cat([blank.unsqueeze(-1), targets.unsqueeze(-1)], -1)
        expanded_targets = expanded_targets.view(N, -1)
        expanded_targets = torch.cat([expanded_targets, blank[:, 0:1]], dim=-1)
        return expanded_targets


    def log_add(self, a, b):
        x, y = torch.max(a, b), torch.min(a, b)
        return x + torch.log1p(torch.exp(y - x))


    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
        log_probs: :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`[NOT IMPLEMENTED YET].
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
	"""
        targets = targets.type(torch.long)
        expanded_targets = self.expand_with_blank(targets)
        N, S = expanded_targets.shape
        T = log_probs.shape[0]

        tiny = torch.finfo().tiny
        # probability of current time step
        #probability = torch.zeros(S, N) + torch.log(torch.tensor(tiny))
        probability = torch.log(torch.zeros(S, N) + tiny)
        probability[0] = log_probs[0, :, self.blank]
        batch_indices = torch.linspace(0, N-1, N).type(torch.long) * log_probs.shape[-1]
        indices = batch_indices + expanded_targets[:, 1]
        probability[1] = log_probs[0].take(indices)

        # (S - 2, N)
        # The previous token can NOT be ignored when the second previous token is identical with
        # specific token.
        mask_skipping = torch.ne(expanded_targets[:, 2:], expanded_targets[:, :-2]).transpose(0, 1)
        mask_skipping = mask_skipping.type(torch.float)
        mask_not_skipping = 1 - mask_skipping
        
        for timestep in range(1, T):
            new_probability1 = self.log_add(probability[1:], probability[:-1])
            new_probability2 = self.log_add(new_probability1[1:], probability[:-2]) * mask_skipping +\
                    new_probability1[1:] * mask_not_skipping
            new_probability = torch.cat([probability[:1],
                new_probability1[:1],
                new_probability2], dim=0)
            probability = new_probability + log_probs[timestep].gather(1, expanded_targets).transpose(0, 1)

            '''
            probability[2:] = torch.log(torch.exp(probability[2:]) +\
                    torch.exp(probability[1:-1]) +\
                    torch.exp(probability[:-2]) * mask_skipping)
            probability[1] = torch.log(torch.exp(probability[0]) + torch.exp(probability[1]) + tiny)
            probability = probability + log_probs[timestep].gather(1, expanded_targets).transpose(0, 1)
            '''
        lengths = (target_lengths * 2 + 1).unsqueeze(0)
        loss = self.log_add(probability.gather(0, lengths - 1), probability.gather(0, lengths - 2))
        #import pdb; pdb.set_trace()
        loss = loss.squeeze(0)
        if self.reduction == 'mean':
            return -(loss / target_lengths.type(torch.float))
        return -loss
