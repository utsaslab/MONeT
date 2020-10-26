#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>

// #include <THC/THCDeviceUtils.cuh>
// #include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
// #include "lrbn.cuh"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

using namespace at;
using namespace at::native;

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> do_forward(
                        const torch::Tensor& input,
                        const torch::Tensor& running_mean,
                        const torch::Tensor& running_var,
                        torch::Tensor& weight,
                        torch::Tensor& bias,
                        bool training,
                        torch::optional<double> momentum,
                        double eps) {
    torch::NoGradGuard no_grad_guard;

    return torch::cudnn_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum.value(),
    eps);
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> do_cudnn_backward(
    const torch::Tensor& input, const torch::Tensor& grad_output,
    const torch::Tensor& weight, const torch::Tensor& running_mean,
    const torch::Tensor& running_var, const torch::Tensor& save_mean,
    const torch::Tensor& save_var, double epsilon,
    const torch::Tensor& reservedSpace) {

//    switch (input.suggest_memory_format()) {
//    case torch::MemoryFormat::Preserve:
//      std::cout << "Preserve\n";
//      break;
//    case torch::MemoryFormat::Contiguous:
//      std::cout << "Contiguous\n";
//      break;
//    case torch::MemoryFormat::ChannelsLast:
//      std::cout << "ChannelsLast\n";
//      break;
//    default:
//      std::cout<<"Unknown memory format\n";
//      break;
//  }

    auto& ctx = torch::globalContext();
    return torch::cudnn_batch_norm_backward(input,
            grad_output.contiguous(input.suggest_memory_format()), weight,
            running_mean, running_var, save_mean, save_var,
            epsilon, reservedSpace);
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> do_native_backward(
    const torch::Tensor& grad_out, const torch::Tensor& self, const torch::Tensor& weight,
    const torch::Tensor& running_mean, const torch::Tensor& running_var,
    const torch::Tensor& save_mean, const torch::Tensor& save_invstd,
    bool train, double epsilon, std::array<bool,3> grad_input_mask) {

    return torch::native_batch_norm_backward(grad_out, self, weight,
            running_mean, running_var, save_mean, save_invstd,
            train, epsilon, grad_input_mask);
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = torch::DefaultPtrTraits, typename index_t = int64_t>
static torch::GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return torch::GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
void batch_norm_bwd1(
    const torch::GenericPackedTensorAccessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t> input,
    const torch::GenericPackedTensorAccessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t> grad_output,
    torch::GenericPackedTensorAccessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t> grad_input,
    torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> grad_weight,
    torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> grad_bias,
    const torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> weight,
    const torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> bias,
    const torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> running_mean,
    const torch::GenericPackedTensorAccessor<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t> running_var,
    const torch::GenericPackedTensorAccessor<stat_accscalar_t, 1, torch::DefaultPtrTraits, index_t> save_mean,
    const torch::GenericPackedTensorAccessor<stat_accscalar_t, 1, torch::DefaultPtrTraits, index_t> save_invstd,
    bool train);

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template1(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
                                                                     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
                                                                     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<stat_scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t>();
  auto grad_output = grad_output_reshaped.generic_packed_accessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t>();
  auto grad_input = packed_accessor_or_dummy<input_scalar_t, 3, torch::DefaultPtrTraits, index_t>(grad_input_reshaped);
  auto weight = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(weight_);
  auto bias = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(bias_);
  auto grad_weight = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(grad_weight_);
  auto grad_bias = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(grad_bias_);
  auto running_mean = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<stat_scalar_t, 1, torch::DefaultPtrTraits, index_t>(running_var_);
  auto save_mean = packed_accessor_or_dummy<accscalar_t, 1, torch::DefaultPtrTraits, index_t>(save_mean_);
  auto save_invstd = packed_accessor_or_dummy<accscalar_t, 1, torch::DefaultPtrTraits, index_t>(save_invstd_);

  // bnback::batch_norm_bwd1<input_scalar_t, stat_scalar_t, accscalar_t, index_t>(input, grad_output, grad_input, grad_weight, grad_bias, weight, bias, running_mean, running_var,
  //    save_mean, save_invstd, train);
  batch_norm_bwd1(input, grad_output, grad_input, grad_weight, grad_bias, weight, bias, running_mean, running_var,
     save_mean, save_invstd, train);

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

std::tuple<Tensor, Tensor, Tensor> output_activated_bn_backward(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
                                                            const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "batch_norm_backward_cuda", [&] {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "batch_norm_backward_cuda", [&] {
      auto mean_st = running_mean.dtype();
      auto var_st = running_var.dtype();
      TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
      // bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
      // bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
      if (at::cuda::detail::canUse32BitIndexMath(self)) {
      //     // return batch_norm_backward_cuda_template1<float, float, int32_t>(grad_out, self, weight, bias, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
          return batch_norm_backward_cuda_template1<scalar_t, scalar_t, int32_t>(grad_out, self, weight, bias, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
          // return batch_norm_backward_cuda_template1<float, float, int64_t>(grad_out, self, weight, bias, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
          return batch_norm_backward_cuda_template1<scalar_t, scalar_t, int64_t>(grad_out, self, weight, bias, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    });
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &do_forward, "BN forward");
    m.def("cudnn_backward", &do_cudnn_backward, "BN backward");
    m.def("native_backward", &do_native_backward, "BN backward");
    m.def("output_activated_bn_backward", &output_activated_bn_backward, "Output activated BN backward");
}
