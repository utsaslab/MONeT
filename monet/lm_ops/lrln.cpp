#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// #include <torch/csrc/autograd/variable.h>

// #include <THC/THCDeviceUtils.cuh>
// #include <THC/THCGeneral.h>
// #include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/core/TensorAccessor.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <c10/macros/Macros.h>
// #include <ATen/cuda/detail/IndexUtils.cuh>
// #include "lrbn.cuh"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

using namespace at;
using namespace at::native;

template <typename T>
void LayerNormForwardCUDA1(
    int64_t M,
    int64_t N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y);

std::tuple<std::tuple<Tensor, Tensor, Tensor>, int64_t, int64_t> do_forward(
    const torch::Tensor& input,
    IntArrayRef normalized_shape,
    const torch::Tensor& weight /* optional */,
    const torch::Tensor& bias /* optional */,
    double eps,
    bool cudnn_enable) {
    torch::NoGradGuard no_grad_guard;

  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();

    return {at::native_layer_norm(X, gamma, beta, M, N, eps), M, N};
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> do_cudnn_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {

    return at::native::layer_norm_backward_cuda(dY, X, mean, rstd, gamma, M, N, grad_input_mask);
}

Tensor layer_norm_recompute_cuda(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    const Tensor& mean1,
    const Tensor& rstd1,
    int64_t M, int64_t N,
    double eps) {

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();
  const auto& mean = mean1.is_contiguous() ? mean1 : mean1.contiguous();
  const auto& rstd = rstd1.is_contiguous() ? rstd1 : rstd1.contiguous();

  Tensor Y = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (M > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "LayerNormKernelImpl1", [&]() {

        DCHECK_EQ(X.numel(), M * N);
        DCHECK(!gamma.defined() || gamma.numel() == N);
        DCHECK(!beta.defined() || beta.numel() == N);
        const scalar_t* X_data = X.data_ptr<scalar_t>();
        const scalar_t* gamma_data = gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr;
        const scalar_t* beta_data = beta.defined() ? beta.data_ptr<scalar_t>() : nullptr;
        scalar_t* Y_data = Y.data_ptr<scalar_t>();
        scalar_t* mean_data = mean.data_ptr<scalar_t>();
        scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
        LayerNormForwardCUDA1<scalar_t>(M, N, X_data, mean_data,
            rstd_data, gamma_data, beta_data, Y_data);
        AT_CUDA_CHECK(cudaGetLastError());
    });
  }
  return std::move(Y);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &do_forward, "LN forward");
    m.def("cudnn_backward", &do_cudnn_backward, "LN backward");
    m.def("forward_recompute", &layer_norm_recompute_cuda, "LN forward recompute");
}