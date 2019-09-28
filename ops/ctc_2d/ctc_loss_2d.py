import torch
from torch.autograd import Function

from . import ctc_2d_csrc


class CTCLoss2DFunction(Function):

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank=0):
        ctx.blank = blank
        if not log_probs.is_cuda:
            raise NotImplementedError

        neg_log_likelihood, log_alpha = ctc_2d_csrc.ctc2d_forward(
            log_probs, targets, input_lengths, target_lengths, blank, torch.finfo().tiny)

        if log_probs.requires_grad:
            ctx.save_for_backward(log_probs, targets, input_lengths,
                                  target_lengths, neg_log_likelihood, log_alpha)
        return neg_log_likelihood

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha = ctx.saved_tensors

        grad_log_probs = torch.ones(2, 3)
        if ctx.needs_input_grad[0]:
            grad_log_probs = ctc_2d_csrc.ctc2d_backward(
                grad_output, log_probs, targets, input_lengths, target_lengths,
                neg_log_likelihood, log_alpha,
                ctx.blank
            )
        return grad_log_probs, None, None, None, None


ctc_loss_2d = CTCLoss2DFunction.apply
