#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cmath>


void print_tsize(at::Tensor t, const char *msg);


std::tuple<at::Tensor, at::Tensor> ctc2d_gpu_template(
    at::Tensor log_probs, at::Tensor targets,
    at::Tensor input_lengths, at::Tensor target_lengths, int64_t BLANK, float TINY
);


at::Tensor ctc2d_gpu_backward_template(
    const at::Tensor grad_out, const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    const at::Tensor neg_log_likelihood, const at::Tensor log_alpha,
    int64_t BLANK
);


std::tuple<at::Tensor, at::Tensor> ctc2d_cuda_forward(
    const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    int64_t BLANK, float TINY
) {
    AT_CHECK(log_probs.is_contiguous(), "log_probs tensor has to be contiguous");

    // shape check
    int64_t batch_size = log_probs.size(2);
    int64_t num_labels = log_probs.size(3);
    AT_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
    AT_CHECK(input_lengths.size(0) == batch_size, "input_lengths must be of size batch_size");
    AT_CHECK(target_lengths.size(0) == batch_size, "target_lengths must be of size batch_size");

    return ctc2d_gpu_template(log_probs, targets, input_lengths, target_lengths, BLANK, TINY);
}


at::Tensor ctc2d_cuda_backward(
    const at::Tensor grad_out, const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    const at::Tensor neg_log_likelihood, const at::Tensor log_alpha,
    int64_t BLANK
) {
    return ctc2d_gpu_backward_template(
        grad_out,
        log_probs, targets, input_lengths, target_lengths,
        neg_log_likelihood, log_alpha,
        BLANK
    );
}