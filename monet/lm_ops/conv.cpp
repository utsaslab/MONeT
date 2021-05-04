#include <torch/extension.h>
#include <mutex>
#include <unordered_map>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

using namespace at;
using namespace at::native;
constexpr int max_dim = 3;

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

static const std::array<cudnnConvolutionFwdAlgo_t, 8> fwd_algos = {
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
};
static const std::array<cudnnConvolutionBwdDataAlgo_t, 6> bwd_algos = {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
};
static const std::array<cudnnConvolutionBwdFilterAlgo_t, 6> bwd_w_algos = {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
};

static inline bool cudnn_conv_use_channels_last(const at::Tensor& input, const at::Tensor& weight) {
  // disable NHWC for float64 input.
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return false;
  }
  return (CUDNN_VERSION >= 7603) &&
      ((input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast));
}

constexpr size_t operator "" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  TORCH_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
           "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    AT_ERROR(ss.str());
  }
}
static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

static inline std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation = IntArrayRef()
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  bool has_dilation = dilation.size() > 0;
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}
static inline std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

torch::Tensor forward_normal(const torch::Tensor& input, const torch::Tensor& weight,
                      torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    static torch::Tensor undefined;
//    return torch::conv2d(input, weight, undefined, stride, padding, dilation, groups);
	return torch::cudnn_convolution(input, weight, undefined, padding, stride, dilation, groups, true, false); //benchmark, deterministic
}
torch::Tensor backward_input_normal(torch::IntArrayRef input_sizes, const torch::Tensor& grad_output_t, const torch::Tensor& weight,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    // torch::Tensor grad_output = grad_output_t.contiguous(weight.suggest_memory_format());
    return torch::cudnn_convolution_backward_input(input_sizes, grad_output_t, weight, padding, stride, dilation, groups, true, false);
}

torch::Tensor backward_weight_normal(torch::IntArrayRef weight_sizes,const torch::Tensor& grad_output_t, const torch::Tensor& input,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    // torch::Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());
    return torch::cudnn_convolution_backward_weight(weight_sizes, grad_output_t, input, padding, stride, dilation, groups, true, false);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {

  memset(params, 0, sizeof(ConvolutionParams));
  // ASSERT(weight.dim() == input.dim())
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i);
    params->input_stride[i] = (int) input.stride(i);
    params->weight_size[i] = (int) weight.size(i);
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};


inline Tensor allocate_workspace(size_t size, const Tensor &other) {
  // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
  // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
  // we manually fail with OOM.
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));
}

// ---------------------------------------------------------------------
//
// Splitting to 32bit
//
// ---------------------------------------------------------------------

template <typename func_t, typename algo_t>
static inline void split_batch_dim_to_32bit_out(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    algo_t algo,
    int64_t max_worksize, func_t func_32bit) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    func_32bit(output, input, weight, padding, stride, dilation, groups, algo);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(max_worksize / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    for (int64_t i = 0; i < num_splits; i++) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor output_ = output.narrow(0, start, split_size_);
      func_32bit(output_, input_, weight, padding, stride, dilation, groups, algo);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}


// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//
void raw_cudnn_convolution_forward_out_32bit(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionFwdAlgo_t algo) {

  auto dataType = CUDNN_DATA_FLOAT;

  ConvolutionArgs args{ input, output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups);
  args.idesc.set(input);
  args.wdesc.set(weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  size_t workspaceSize;
  cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        &workspaceSize);

  Tensor workspace = allocate_workspace(workspaceSize, input);

  // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
  // whether to use Tensor core kernels or not
  // See Note [behavior of cudnnFind and cudnnGet]
  AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), CUDNN_DEFAULT_MATH));

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input.data_ptr(),
    args.wdesc.desc(), weight.data_ptr(),
    args.cdesc.desc(), algo, workspace.data_ptr(), workspaceSize,
    &zero, args.odesc.desc(), output.data_ptr()));
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionFwdAlgo_t algo) {
  split_batch_dim_to_32bit_out(output, input, weight, padding, stride, dilation, groups, algo, 1024 * 1024 * 256, raw_cudnn_convolution_forward_out_32bit);
}

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionFwdAlgo_t algo)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto layout = cudnn_conv_use_channels_last(*input, *weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto output_t = at::empty(
                    conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation),
                    input->options(),
                    layout);

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(layout);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), layout);
  Tensor input_contig = input->contiguous(layout);
  input_contig.resize_(input_contig.sizes(), layout);

  raw_cudnn_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, algo);

  return *output;
}

// same as cudnn_convolution_transpose_backward_input_2
Tensor cudnn_convolution_2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, int alg_type)
{
  const cudnnConvolutionFwdAlgo_t algo = (0 <= alg_type && alg_type < fwd_algos.size()) ? fwd_algos[alg_type] : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  auto output_t = cudnn_convolution_forward(
    "cudnn_convolution", input, weight, padding, stride, dilation, groups, algo);
  return output_t;
}


// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_input_out_32bit(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdDataAlgo_t algo) {
  auto dataType = CUDNN_DATA_FLOAT;

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups);
  args.idesc.set(grad_input);
  args.wdesc.set(weight, 0, grad_output.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);
  size_t workspaceSize;
  AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            &workspaceSize));
  Tensor workspace = allocate_workspace(workspaceSize, grad_output);

  // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
  // whether to use Tensor core kernels or not
  // See Note [behavior of cudnnFind and cudnnGet]
  AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), CUDNN_DEFAULT_MATH));

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardData(
      args.handle,
      &one, args.wdesc.desc(), weight.data_ptr(),
      args.odesc.desc(), grad_output.data_ptr(),
      args.cdesc.desc(), algo, workspace.data_ptr(), workspaceSize,
      &zero, args.idesc.desc(), grad_input.data_ptr()));
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdDataAlgo_t algo) {
  split_batch_dim_to_32bit_out(grad_input, grad_output, weight, padding, stride, dilation, groups, algo, 1024 * 1024 * 128, raw_cudnn_convolution_backward_input_out_32bit);
}

// NOTE [ Backward vs transpose convolutions ]
//
// Backward and transpose are algorithmically equivalent, but they
// compute their geometry differently.  In a backwards, you knew what
// the original size of the input tensor was, so you can cache that
// geometry and fill it directly.  In transposed convolution, it is
// more conventional to not explicitly specify the output (previously
// input) size, and compute it.  This, however, leaves a degree of
// freedom; this degree of freedom is resolved using the
// output_padding parameter.  Both of these interfaces are equivalent,
// but they are differently convenient depending on the use case.

Tensor cudnn_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdDataAlgo_t algo)
{
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto layout = cudnn_conv_use_channels_last(*grad_output, *weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto grad_input_t = at::empty(input_size, grad_output->options(), layout);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(layout);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), layout);

  Tensor grad_output_contig = grad_output->contiguous(layout);
  grad_output_contig.resize_(grad_output_contig.sizes(), layout);

  raw_cudnn_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, algo);

  return *grad_input;
}
Tensor cudnn_convolution_backward_input_2(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    int alg_type)
{
  const cudnnConvolutionBwdDataAlgo_t algo = (0 <= alg_type && alg_type < bwd_algos.size()) ? bwd_algos[alg_type] : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return cudnn_convolution_backward_input(
      "cudnn_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, algo);
}


Tensor cudnn_convolution_transpose_forward(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdDataAlgo_t algo)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return cudnn_convolution_backward_input(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, algo);
}

Tensor cudnn_convolution_transpose_2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, int alg_type)
{
  const cudnnConvolutionBwdDataAlgo_t algo = (0 <= alg_type && alg_type < bwd_algos.size()) ? bwd_algos[alg_type] : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "cudnn_convolution_transpose";
  auto output_t = cudnn_convolution_transpose_forward(
    c, input, weight, padding, output_padding, stride, dilation, groups, algo);
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdFilterAlgo_t algo) {

  auto dataType = CUDNN_DATA_FLOAT;

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups);
  args.idesc.set(input);
  args.wdesc.set(grad_weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  size_t workspaceSize;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        &workspaceSize);

  Tensor workspace = allocate_workspace(workspaceSize, input);

  // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
  // whether to use Tensor core kernels or not
  // See Note [behavior of cudnnFind and cudnnGet]
  AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), CUDNN_DEFAULT_MATH));

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      args.handle,
      &one, args.idesc.desc(), input.data_ptr(),
      args.odesc.desc(), grad_output.data_ptr(),
      args.cdesc.desc(), algo, workspace.data_ptr(), workspaceSize,
      &zero, args.wdesc.desc(), grad_weight.data_ptr()));
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdFilterAlgo_t algo) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = grad_output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    raw_cudnn_convolution_backward_weight_out_32bit(grad_weight, grad_output, input, padding, stride, dilation, groups, algo);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = grad_output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(1024 * 1024 * 512 / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    for (int64_t i = 0; i < num_splits; i++) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
      Tensor grad_weight_ = at::empty_like(grad_weight);
      raw_cudnn_convolution_backward_weight_out_32bit(grad_weight_, grad_output_, input_, padding, stride, dilation, groups, algo);
      grad_weight.add_(grad_weight_);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

Tensor cudnn_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const Tensor& grad_output_t, const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    cudnnConvolutionBwdFilterAlgo_t algo)
{
  auto layout = cudnn_conv_use_channels_last(input_t, grad_output_t) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  Tensor grad_output_contig_t = grad_output_t.contiguous(layout);
  // Make sure that NC11 strides follow formula
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), layout);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  Tensor input_contig_t = input_t.contiguous(layout);
  input_contig_t.resize_(input_contig_t.sizes(), layout);
  TensorArg input{ input_contig_t, "input", 2};

  checkAllSameType(c, {grad_output_contig, input});
  checkAllSameGPU(c, {grad_output_contig, input});

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), layout);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  raw_cudnn_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input,
      padding, stride, dilation, groups, algo);

  return grad_weight_t;
}

Tensor cudnn_convolution_backward_weight_2(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    int alg_type)
{
  const cudnnConvolutionBwdFilterAlgo_t algo = (0 <= alg_type && alg_type < bwd_w_algos.size()) ? bwd_w_algos[alg_type] : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, grad_output_t, input_t,
      padding, stride, dilation, groups, algo);
}

Tensor cudnn_convolution_transpose_backward_weight_2(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    int alg_type)
{
  const cudnnConvolutionBwdFilterAlgo_t algo = (0 <= alg_type && alg_type < bwd_w_algos.size()) ? bwd_w_algos[alg_type] : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, input_t, grad_output_t,
      padding, stride, dilation, groups, algo);
}

Tensor convolution_main(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  if (bias.dim() == 0) {
    return at::convolution(input, weight, Tensor(), stride, padding, dilation, transposed, output_padding, groups);
  } else {
    return at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
}

std::tuple<Tensor,Tensor> backward_depthwise(
  const Tensor & grad_output,
  const Tensor & self,
  const Tensor & weight,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  std::array<bool,2> output_mask
) {
  return torch::thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("n_fwd_algos", [](){return fwd_algos.size();});
    m.def("n_bwd_ip_algos", [](){return bwd_algos.size();});
    m.def("n_bwd_wt_algos", [](){return bwd_w_algos.size();});
    m.def("cudnn_convolution", &cudnn_convolution_2);
    m.def("cudnn_convolution_backward_input", &cudnn_convolution_backward_input_2);
    m.def("cudnn_convolution_backward_weight", &cudnn_convolution_backward_weight_2);

    m.def("cudnn_convolution_transpose", &cudnn_convolution_transpose_2);
    m.def("cudnn_convolution_transpose_backward_input", &cudnn_convolution_2);
    m.def("cudnn_convolution_transpose_backward_weight", &cudnn_convolution_transpose_backward_weight_2);

    m.def("forward_normal", &forward_normal, "Conv forward");
    m.def("backward_input_normal", &backward_input_normal, "Conv backward input");
    m.def("backward_weight_normal", &backward_weight_normal, "Conv backward weight");
    m.def("convolution_main", &convolution_main, "First conv abstraction");
    m.def("backward_depthwise", &backward_depthwise, "Conv backward thnn depthwise");
}
