#pragma once

#ifdef WITH_CUDA
#include "cuda/ctc2d.h"
#endif

std::tuple<at::Tensor, at::Tensor> ctc2d_forward(
    at::Tensor log_probs, at::Tensor targets, at::Tensor input_lengths, at::Tensor target_lengths,
    int64_t BLANK, float TINY
) {
    if (log_probs.type().is_cuda()) {
#ifdef WITH_CUDA
    return ctc2d_cuda_forward(
        log_probs, targets, input_lengths, target_lengths, BLANK, TINY
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor ctc2d_backward(
    at::Tensor grad_out,
    at::Tensor log_probs, at::Tensor targets, at::Tensor input_lengths, at::Tensor target_lengths,
    at::Tensor neg_log_likelihood, at::Tensor log_alpha,
    int64_t BLANK
) {
    if (log_probs.type().is_cuda()) {
#ifdef WITH_CUDA
    return ctc2d_cuda_backward(
        grad_out,
        log_probs, targets, input_lengths, target_lengths,
        neg_log_likelihood, log_alpha,
        BLANK
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}