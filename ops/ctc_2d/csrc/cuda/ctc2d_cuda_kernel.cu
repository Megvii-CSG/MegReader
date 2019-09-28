#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <numeric>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

void print_tsize(at::Tensor t, const char *msg) {
    printf("%s size: ");
    for (int i = 0; i < t.ndimension(); i++) {
        printf("%d ", int(t.size(i)));
    }
    printf("\n");
}


// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1]) note that no bound-checking is done
// __restrict__ impact to be measured, https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
template <typename scalar_t>
__device__ static inline int64_t get_target_prime(
    const scalar_t* __restrict__ target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK
) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

template <typename scalar_t>
__device__ static inline scalar_t safe_log_add(scalar_t a, scalar_t b)
{
    scalar_t m=((a > b) ? a : b);
    if (m == -INFINITY)
        m = 0;
    return (std::log(std::exp(a-m) + std::exp(b-m)) + m);
}


template <typename scalar_t>
__global__ void ctc2d_log_alpha_gpu_kernel(
    const int64_t n,
    scalar_t* __restrict__ log_alpha_data, const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths, int max_input_length,
    const int64_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths, const int max_target_length,
    scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t lp_input_stride, int64_t lp_height_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_height_stride, int64_t la_target_stride,
    int64_t tg_batch_stride, int64_t tg_target_stride,
    int64_t batch_size, int64_t batch_per_block, int64_t height, int64_t BLANK
) {
  CUDA_KERNEL_LOOP(index, n)
  {
    int64_t b = (index - blockIdx.x*blockDim.x) / (2*max_target_length+1) + blockIdx.x*batch_per_block;
    int64_t s = (index - blockIdx.x*blockDim.x) % (2*max_target_length+1);

    if ((b >= batch_size) || (b >= (blockIdx.x+1)*batch_per_block))
        return;

    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    // log_probs_data ==> [T, H, N, C]
    // log_alpha_data ==> [N, T, H, 2*S+1]
    int64_t lp_batch_offset = b*lp_batch_stride;
    int64_t la_batch_offset = b*la_batch_stride;
    int64_t tg_batch_offset = b*tg_batch_stride;

    scalar_t la;
    switch (s) {
    case 0:
        for (int64_t h=0; h < height; h++) {
            la = log_probs_data[lp_height_stride*h + lp_batch_offset + lp_char_stride*BLANK];
            if (s < 2*max_target_length+1)
                log_alpha_data[la_batch_offset + la_height_stride*h + la_target_stride*s] = la;
        }
        break;
    case 1:
        for (int64_t h=0; h < height; h++) {
            if (target_length > 0) {
                la = log_probs_data[lp_height_stride*h + lp_batch_offset +
                    lp_char_stride*get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];
            }
            else {
                la = -INFINITY;
            }
            if (s < 2*max_target_length+1)
                log_alpha_data[la_batch_offset + la_height_stride*h + la_target_stride*s] = la;
        }
        break;
    default:
        la = -INFINITY;
        if (s < 2*max_target_length+1) {
            for (int64_t h=0; h < height; h++)
                log_alpha_data[la_batch_offset + la_height_stride*h + la_target_stride*s] = la;
        }
    }

    // These two only depend on s, so we can cache them.
    int64_t current_char;       // l_s in eq (6)
    bool have_three;            // flag which of the two cases in eq (6) we have
    if (s < 2*target_length+1) {
        current_char = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
        have_three = ((s > 1) &&
            (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) != current_char));
    } else {
        current_char = BLANK;
        have_three = false;
    }

    for (int64_t t=1; t < max_input_length; t++) {
        // on cuda 9 we might use partial synchronization of only the threads within the same batch
        __syncthreads();
        if ((t < input_length) && (target_length > 0) && (s < 2*target_length+1)) {
            // only for valid t, s. This is equation (6) and (7), la1, la2, la3 are the three summands,
            // lamax is the maximum for the logsumexp trick.
            scalar_t la1 = log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                    la_height_stride*0 + la_target_stride*s];
            for (int64_t h=1; h < height; h++) {
                la1 = safe_log_add(la1, log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                    la_height_stride*h + la_target_stride*s]);
            }
            scalar_t lamax = la1;
            scalar_t la2, la3;
            if (s > 0) {
                la2 = log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                        la_height_stride*0 + la_target_stride*(s-1)];
                for (int64_t h=1; h < height; h++) {
                    la2 = safe_log_add(la2, log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                        la_height_stride*h + la_target_stride*(s-1)]);
                }
                if (la2 > lamax)
                    lamax = la2;
            } else {
                la2 = -INFINITY;
            }

            if (have_three) {
                la3 = log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                            la_height_stride*0 + la_target_stride*(s-2)];
                for (int64_t h=1; h < height; h++) {
                    la3 = safe_log_add(la3, log_alpha_data[la_batch_offset + la_input_stride*(t-1) +
                            la_height_stride*h + la_target_stride*(s-2)]);
                }
                if (la3 > lamax)
                    lamax = la3;
            } else {
                la3 = -INFINITY;
            }

            // when all are neginf. (then the whole thing is neginf, but we can pretend)
            if (lamax == -INFINITY)
                lamax = 0;

            for (int64_t h=0; h < height; h++) {
                log_alpha_data[la_batch_offset + la_input_stride*t + la_height_stride*h + la_target_stride*s] =
                    std::log(std::exp(la1-lamax) + std::exp(la2-lamax) + std::exp(la3-lamax)) + lamax +
                    log_probs_data[lp_input_stride*t + lp_height_stride*h +
                    lp_batch_offset + lp_char_stride*current_char];
            }
        } else {
            // otherwise we just set to neginf
            if (s < 2*max_target_length+1) {
                for (int64_t h = 0; h < height; h++) {
                    log_alpha_data[la_batch_offset + la_input_stride * t +
                        la_height_stride * h + la_target_stride * s] = -INFINITY;
                }
            }
        }
    }
    // on cuda 9 we might use partial synchronization of only the threads within the same batch
    __syncthreads();

    // compute the loss (eq (8))
    if (s == 0) {
        scalar_t l1 = log_alpha_data[la_batch_offset + la_input_stride*(input_length-1) +
                la_height_stride*0 + la_target_stride*(target_length*2)];
        for (int64_t h=1; h < height; h++) {
            l1 = safe_log_add(l1, log_alpha_data[la_batch_offset + la_input_stride*(input_length-1) +
                la_height_stride*h + la_target_stride*(target_length*2)]);
        }

        scalar_t l2 = log_alpha_data[la_batch_offset + la_input_stride*(input_length-1) +
                la_height_stride*0 + la_target_stride*(target_length*2-1)];
        for (int64_t h=1; h < height; h++) {
            l2 = safe_log_add(l2, log_alpha_data[la_batch_offset + la_input_stride*(input_length-1) +
                la_height_stride*h + la_target_stride*(target_length*2-1)]);
        }

        scalar_t m = ((l1 > l2) ? l1 : l2);
        if (m == -INFINITY)
            m = 0;
        scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
        neg_log_likelihood_data[b] = -log_likelihood;
    }
  }
}


std::tuple<at::Tensor, at::Tensor> ctc2d_gpu_template(
    const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    int64_t BLANK, float TINY
) {
    int64_t max_target_length = targets.size(1);
    AT_CHECK((2 * max_target_length + 1) <= CUDA_NUM_THREADS, "max target length out of range, got ", max_target_length,
        ", must less than ", CUDA_NUM_THREADS);

    int64_t max_input_length = log_probs.size(0);
    int64_t height = log_probs.size(1);
    int64_t batch_size = log_probs.size(2);
    int64_t batch_per_block = CUDA_NUM_THREADS / (2 * max_target_length + 1);

    const int num_kernels = (batch_size + batch_per_block - 1) / batch_per_block * CUDA_NUM_THREADS;
    // N T H 2*S+1
    at::Tensor log_alpha = at::zeros(
        {batch_size, log_probs.size(0), log_probs.size(1), 2*max_target_length+1},
        log_probs.options()
    );
    at::Tensor neg_log_likelihood = at::zeros({batch_size}, log_probs.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.type(), "ctc2d_log_alpha_gpu_template", ([&] {
        ctc2d_log_alpha_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            log_alpha.data<scalar_t>(), log_probs.data<scalar_t>(),
            input_lengths.data<int64_t>(), max_input_length,
            targets.data<int64_t>(),
            target_lengths.data<int64_t>(), max_target_length,
            neg_log_likelihood.data<scalar_t>(),
            log_probs.stride(0), log_probs.stride(1), log_probs.stride(2), log_probs.stride(3),
            log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2), log_alpha.stride(3),
            targets.stride(0), targets.stride(1),
            batch_size, batch_per_block, height, BLANK
        );
      }));
    return std::make_tuple(neg_log_likelihood, log_alpha);
}


template <typename scalar_t>
__global__ void ctc2d_log_beta_gpu_kernel(
    const int64_t n,
    scalar_t* __restrict__ log_beta_data, const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths, int max_input_length,
    const int64_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths, const int max_target_length,
    int64_t lp_input_stride, int64_t lp_height_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_height_stride, int64_t lb_target_stride,
    int64_t tg_batch_stride, int64_t tg_target_stride,
    int64_t batch_size, int64_t batch_per_block, int64_t height, int64_t BLANK
) {
  CUDA_KERNEL_LOOP(index, n)
  {
    int64_t b = (index - blockIdx.x*blockDim.x) / (2*max_target_length+1) + blockIdx.x*batch_per_block;
    int64_t s = (index - blockIdx.x*blockDim.x) % (2*max_target_length+1);

    if ((b >= batch_size) || (b >= (blockIdx.x+1)*batch_per_block))
        return;

    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    // log_probs_data ==> [T, H, N, C]
    // log_beta_data ==> [N, T, H, 2*S+1]
    int64_t lp_batch_offset = b*lp_batch_stride;
    int64_t lb_batch_offset = b*lb_batch_stride;
    int64_t tg_batch_offset = b*tg_batch_stride;

    scalar_t lb;
    if (s == 2*target_length) {
      for (int64_t h=0; h < height; h++) {
        lb = log_probs_data[lp_input_stride*(input_length-1) + lp_height_stride*h + lp_batch_offset + lp_char_stride*BLANK];
        log_beta_data[lb_batch_offset + lb_input_stride*(input_length-1) + lb_height_stride*h + lb_target_stride*s] = lb;
      }
    } else if ((target_length > 0) && (s == 2*target_length-1)) {
      int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      for (int64_t h=0; h < height; h++) {
        lb = log_probs_data[lp_input_stride*(input_length-1) + lp_height_stride*h + lp_batch_offset +
          lp_char_stride*current_target_prime];
        log_beta_data[lb_batch_offset + lb_input_stride*(input_length-1) + lb_height_stride*h + lb_target_stride*s] = lb;
      }
    } else {
      for (int64_t h=0; h < height; h++) {
        log_beta_data[lb_batch_offset + lb_input_stride*(input_length-1) + lb_height_stride*h + lb_target_stride*s] =
          -INFINITY;
      }
    }

    int64_t current_target_prime;
    bool have_three;
    if (s < 2*target_length+1) {
      current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      have_three = ((s < 2*target_length-1) &&
                    (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) !=
                     current_target_prime));
    } else {
      current_target_prime = BLANK;
      have_three = false;
    }

    // now go backward in t. Note that we need to skip the last timestep that we did above.
    for (int64_t t=max_input_length-2; t>=0; t--) {
      __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch item
      if ((t < input_length-1) && (target_length > 0) && (s < 2*target_length+1)) {
        scalar_t lb1 = log_beta_data[lb_batch_offset + lb_input_stride*(t+1) +
                    lb_height_stride*0 + lb_target_stride*s];
        for (int64_t h=1; h < height; h++) {
            lb1 = safe_log_add(lb1, log_beta_data[lb_batch_offset + lb_input_stride*(t+1) +
                lb_height_stride*h + lb_target_stride*s]);
        }
        scalar_t lbmax = lb1;
        scalar_t lb2, lb3;

        if (s < 2*target_length) {
          lb2 = log_beta_data[
            lb_batch_offset + lb_input_stride*(t+1) + lb_height_stride*0 + lb_target_stride*(s+1)];
          for (int64_t h=1; h < height; h++) {
            lb2 = safe_log_add(lb2, log_beta_data[lb_batch_offset + lb_input_stride*(t+1) +
              lb_height_stride*h + lb_target_stride*(s+1)]);
          }
          if (lb2 > lbmax)
            lbmax = lb2;
        } else {
          lb2 = -INFINITY;
        }

        if (have_three) {
          lb3 = log_beta_data[lb_batch_offset + lb_input_stride*(t+1) +
            lb_height_stride*0 + lb_target_stride*(s+2)];
          for (int64_t h=1; h < height; h++) {
            lb3 = safe_log_add(lb3, log_beta_data[lb_batch_offset + lb_input_stride*(t+1) +
              lb_height_stride*h + lb_target_stride*(s+2)]);
          }
          if (lb3 > lbmax)
            lbmax = lb3;
        } else {
            lb3 = -INFINITY;
        }
        if (lbmax == -INFINITY)
          lbmax = 0;

        scalar_t tmp = std::log(std::exp(lb1-lbmax) + std::exp(lb2-lbmax) + std::exp(lb3-lbmax)) + lbmax;
        for (int64_t h=0; h < height; h++) {
          log_beta_data[lb_batch_offset + lb_input_stride*t + lb_height_stride*h + lb_target_stride*s] =
            tmp + log_probs_data[lp_input_stride*t + lp_height_stride*h +
              lp_batch_offset + lp_char_stride*current_target_prime];
        }
      } else if ((target_length == 0) || (s > 2*target_length+1) || (t >= input_length)) {
        for (int64_t h=0; h < height; h++) {
          log_beta_data[lb_batch_offset + lb_input_stride*t + lb_height_stride*h + lb_target_stride*s] = -INFINITY;
        }
      }
    }
  }
}


template <typename scalar_t>
__global__ void ctc2d_backward_collect_nonblank_gpu_kernel(
    const int64_t n,
    scalar_t* __restrict__ gradient_data,
    const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
    const scalar_t* __restrict__ log_alpha_data, const scalar_t* __restrict__ log_beta_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths, int64_t max_input_length,
    const int64_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths, int64_t max_target_length,
    const scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t gr_input_stride, int64_t gr_height_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
    int64_t lp_input_stride, int64_t lp_height_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_height_stride, int64_t la_target_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_height_stride, int64_t lb_target_stride,
    int64_t tg_batch_stride, int64_t tg_target_stride,
    int64_t batch_size, int64_t batch_per_block, int64_t height, int64_t BLANK, bool zero_infinity
) {
  CUDA_KERNEL_LOOP(index, n)
  {
    int64_t b = (index - blockIdx.x*blockDim.x) / max_target_length + blockIdx.x*batch_per_block;
    int64_t s = (index - blockIdx.x*blockDim.x) % max_target_length;
    if (b >= batch_size)
      return;

    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t gr_batch_offset = b*gr_batch_stride;
    int64_t lp_batch_offset = b*lp_batch_stride;
    int64_t la_batch_offset = b*la_batch_stride;
    int64_t lb_batch_offset = b*lb_batch_stride;
    int64_t tg_batch_offset = b*tg_batch_stride;

    if (s >= target_length)
      return;

    int64_t target = targets_data[tg_batch_offset + s*tg_target_stride];
    scalar_t nll = neg_log_likelihood_data[b];
    scalar_t gr =  grad_out_data[b * grad_out_batch_stride];

    if (zero_infinity && nll == INFINITY)
      return;

    for (int64_t t = 0; t < input_length; t++) {
      for (int64_t h = 0; h < height; h++) {
        scalar_t lp = log_probs_data[lp_batch_offset + lp_input_stride*t + lp_height_stride*h + lp_char_stride*target];
        atomicAdd(&gradient_data[gr_batch_offset + gr_input_stride*t + gr_height_stride*h + gr_char_stride*target],
          -std::exp(log_alpha_data[la_batch_offset + la_input_stride*t + la_height_stride*h + la_target_stride*(s*2+1)]
          + log_beta_data[lb_batch_offset + lb_input_stride*t + lb_height_stride*h + lb_target_stride*(s*2+1)]
          + nll - lp) * gr);
      }
    }
  }
}


template <typename scalar_t>
__global__ void ctc2d_backward_collect_gpu_kernel(
    const int64_t n,
    scalar_t* __restrict__ gradient_data,
    const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
    const scalar_t* __restrict__ log_alpha_data, const scalar_t* __restrict__ log_beta_data,
    const scalar_t* log_probs_data,
    const int64_t* __restrict__ input_lengths, int64_t max_input_length,
    const int64_t* __restrict__ targets_data,
    const int64_t* __restrict__ target_lengths, int64_t max_target_length,
    const scalar_t* __restrict__ neg_log_likelihood_data,
    int64_t gr_input_stride, int64_t gr_height_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
    int64_t lp_input_stride, int64_t lp_height_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_height_stride, int64_t la_target_stride,
    int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_height_stride, int64_t lb_target_stride,
    int64_t tg_batch_stride, int64_t tg_target_stride,
    int64_t batch_size, int64_t num_labels, int64_t batch_per_block, int64_t height, int64_t BLANK, bool zero_infinity
) {
  CUDA_KERNEL_LOOP(index, n)
  {
    int64_t b = (index - blockIdx.x*blockDim.x) / max_input_length + blockIdx.x*batch_per_block;
    int64_t t = (index - blockIdx.x*blockDim.x) % max_input_length;
    if (b >= batch_size)
      return;

    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t gr_batch_offset = b*gr_batch_stride;
    int64_t lp_batch_offset = b*lp_batch_stride;
    int64_t la_batch_offset = b*la_batch_stride;
    int64_t lb_batch_offset = b*lb_batch_stride;
    int64_t tg_batch_offset = b*tg_batch_stride;

    // collected[b, t, h, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
    for (int64_t s = 0; s < 2*max_target_length+1; s++) {
      if ((target_length > 0) && (s < 2*target_length+1)) {
        int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
        /*scalar_t laaa = log_alpha_data[la_batch_offset + la_input_stride*t +
          la_height_stride*0 + la_target_stride*s];
        for (int64_t h = 1; h < height; h++) {
          laaa = safe_log_add(laaa, log_alpha_data[la_batch_offset + la_input_stride*t +
                la_height_stride*h + la_target_stride*s]);
        }

        scalar_t lbbb = log_beta_data[lb_batch_offset + lb_input_stride*t +
          lb_height_stride*0 + lb_target_stride*s];
        for (int64_t h = 1; h < height; h++) {
          lbbb = safe_log_add(lbbb, log_beta_data[lb_batch_offset + lb_input_stride*t +
                lb_height_stride*h + lb_target_stride*s]);
        }*/

        for (int64_t h = 0; h < height; h++) {
          scalar_t laaa = log_alpha_data[la_batch_offset + la_input_stride*t +
            la_height_stride*h + la_target_stride*s];
          scalar_t lbbb = log_beta_data[lb_batch_offset + lb_input_stride*t +
            lb_height_stride*h + lb_target_stride*s];

          scalar_t log_alpha_beta = laaa + lbbb;
          scalar_t& lcab =
            gradient_data[gr_batch_offset + gr_input_stride*t + gr_height_stride*h + gr_char_stride*current_target_prime];

          if (lcab == -INFINITY) {
            lcab = log_alpha_beta;
          } else {
            scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
            lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta-max))+max;
          }
        }
      }
    }

    scalar_t nll = neg_log_likelihood_data[b];
    scalar_t gr =  grad_out_data[b * grad_out_batch_stride];

    for (int64_t c = 0; c < num_labels; c++) {
      for (int64_t h = 0; h < height; h++) {
        scalar_t& res = gradient_data[gr_batch_offset + gr_input_stride*t + gr_height_stride*h + gr_char_stride*c];
        if (t < input_length && (! zero_infinity || nll != INFINITY)) {
          scalar_t lp = log_probs_data[lp_batch_offset + lp_input_stride*t + lp_height_stride*h + lp_char_stride*c];
          if (res == -INFINITY)
            res = 0;
          else
            res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
        }
        else {
          res = 0.;
        }
      }
    }
  }
}


at::Tensor ctc2d_gpu_backward_template(
    const at::Tensor grad_out, const at::Tensor log_probs, const at::Tensor targets,
    const at::Tensor input_lengths, const at::Tensor target_lengths,
    const at::Tensor neg_log_likelihood, const at::Tensor log_alpha,
    int64_t BLANK
) {
    bool zero_infinity = 0;
    int64_t max_target_length = targets.size(1);

    int64_t max_input_length = log_probs.size(0);
    int64_t height = log_probs.size(1);
    int64_t batch_size = log_probs.size(2);
    int64_t num_labels = log_probs.size(3);

    int64_t batch_per_block = CUDA_NUM_THREADS / (2 * max_target_length + 1);
    int64_t num_kernels = (batch_size + batch_per_block - 1) / batch_per_block * CUDA_NUM_THREADS;
    at::Tensor log_beta = at::zeros(
        {batch_size, log_probs.size(0), log_probs.size(1), 2*max_target_length+1},
        log_probs.options()
    );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.type(), "ctc2d_log_beta_gpu_template", ([&] {
        ctc2d_log_beta_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            log_beta.data<scalar_t>(), log_probs.data<scalar_t>(),
            input_lengths.data<int64_t>(), max_input_length,
            targets.data<int64_t>(),
            target_lengths.data<int64_t>(), max_target_length,
            log_probs.stride(0), log_probs.stride(1), log_probs.stride(2), log_probs.stride(3),
            log_beta.stride(0), log_beta.stride(1), log_beta.stride(2), log_beta.stride(3),
            targets.stride(0), targets.stride(1),
            batch_size, batch_per_block, height, BLANK
        );
    }));

    at::Tensor grad = at::full_like(log_probs, -INFINITY);
    // bool is_large = (2*log_probs.size(0)+(24*batch_size)/10+(2*num_labels)/10) > 450;
    bool is_large = 0;
    if (is_large) { // large alphabet, large batch
        // std::cout << "+++Large+++" << std::endl;
        // this computes the probs, minuend in (16)
        exp_out(grad, log_probs);

        // now we compute the subtrahend for the blanks. It is a straightforward reduction because we know that
        // blanks are in every other position.
        // maybe we should kernelize this, too.
        auto grad_blank = grad.narrow(3, BLANK, 1);
        grad_blank -= (at::logsumexp(
            log_alpha.as_strided({batch_size, log_alpha.size(1), log_alpha.size(2), max_target_length+1},
                {log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2), log_alpha.stride(3)*2}) +
            log_beta.as_strided({batch_size, log_beta.size(1), log_beta.size(2), max_target_length+1},
                {log_beta.stride(0), log_beta.stride(1), log_beta.stride(2), log_beta.stride(3)*2}),
            3, true)
            .permute({1, 2, 0, 3})
            .add_(neg_log_likelihood.view({1, 1, batch_size, 1}))
            .sub_(log_probs.narrow(3, BLANK, 1))
            .exp_()
        );
        grad *= grad_out.view({1, 1, batch_size, 1});

        // For the non-blank characters, we use a kernel to compute the subtrahend.
        // Again we might configure block and grid in a better way.
        batch_per_block = CUDA_NUM_THREADS / max_target_length;
        num_kernels = (batch_size + batch_per_block - 1) / batch_per_block * CUDA_NUM_THREADS;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.type(), "ctc2d_collect_nonblank", ([&] {
        ctc2d_backward_collect_nonblank_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            grad.data<scalar_t>(),
            grad_out.data<scalar_t>(), grad_out.stride(0),
            log_alpha.data<scalar_t>(), log_beta.data<scalar_t>(),
            log_probs.data<scalar_t>(),
            input_lengths.data<int64_t>(), max_input_length,
            targets.data<int64_t>(),
            target_lengths.data<int64_t>(), max_target_length,
            neg_log_likelihood.data<scalar_t>(),
            grad.stride(0), grad.stride(1), grad.stride(2), grad.stride(3),
            log_probs.stride(0), log_probs.stride(1), log_probs.stride(2), log_probs.stride(3),
            log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2), log_alpha.stride(3),
            log_beta.stride(0), log_beta.stride(1), log_beta.stride(2), log_beta.stride(3),
            targets.stride(0), targets.stride(1),
            batch_size, batch_per_block, height, BLANK, zero_infinity
        );
        }));
    } else { // small problem, use naive algorithm
        // std::cout << "+++Small+++" << std::endl;
        batch_per_block = CUDA_NUM_THREADS / max_input_length;
        num_kernels = (batch_size + batch_per_block - 1) / batch_per_block * CUDA_NUM_THREADS;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.type(), "ctc2d_collect_all", ([&] {
        ctc2d_backward_collect_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            grad.data<scalar_t>(),
            grad_out.data<scalar_t>(), grad_out.stride(0),
            log_alpha.data<scalar_t>(), log_beta.data<scalar_t>(),
            log_probs.data<scalar_t>(),
            input_lengths.data<int64_t>(), max_input_length,
            targets.data<int64_t>(),
            target_lengths.data<int64_t>(), max_target_length,
            neg_log_likelihood.data<scalar_t>(),
            grad.stride(0), grad.stride(1), grad.stride(2), grad.stride(3),
            log_probs.stride(0), log_probs.stride(1), log_probs.stride(2), log_probs.stride(3),
            log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2), log_alpha.stride(3),
            log_beta.stride(0), log_beta.stride(1), log_beta.stride(2), log_beta.stride(3),
            targets.stride(0), targets.stride(1),
            batch_size, num_labels, batch_per_block, height, BLANK, zero_infinity
        );
        }));
    }
    return grad;
}
