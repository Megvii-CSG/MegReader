#pragma once
#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor> ctc2d_cuda_forward(
    const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    int64_t BLANK, float TINY
);


at::Tensor ctc2d_cuda_backward(
    const at::Tensor grad_out,
    const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    const at::Tensor neg_log_likelihood, const at::Tensor log_alpha,
    int64_t BLANK
);