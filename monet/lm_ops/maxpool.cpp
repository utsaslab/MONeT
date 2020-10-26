#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

static const int BLOCK_THREADS = 256;

void max_pool_forward_nchw_cuda(const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, float* top_data,
    uint8_t* top_mask);

void max_pool_backward_nchw_cuda(const int nthreads, const float* top_diff,
    const uint8_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    float* bottom_diff);

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit

#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

// AveragePool2d/DilatedMaxPool2d (forward)
template <typename dest_t, typename src_t>
static inline dest_t
safe_downcast(src_t v)
{
  TORCH_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(),
              "integer out of range");

  return static_cast<dest_t>(v);
}

template<typename T>
static inline T div_rtn(T x, T y) {
    int q = x/y;
    int r = x%y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return q;
}

template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (pad_l) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l)
          --outputSize;
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

static inline void
pool2d_shape_check(
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth,
  int64_t nnumel, int64_t ndim)
{
//  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got "
              "dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationH > 0 && dilationW > 0,
              "dilation should be greater than zero, but got ",
              "dilationH: ", dilationH, " dilationW: ", dilationW);

  TORCH_CHECK(nnumel > 0 && (ndim == 3 || ndim == 4),
              "non-empty 3D or 4D input tensor expected but got ndim: ", ndim);
  TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
              "pad should be smaller than half of kernel size, but got ",
              "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size: (",
              nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
              "Calculated output size: (",
              nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
              "Output size is too small");
}

// DilatedMaxPool2d (backward)
static inline void
max_pool2d_backward_shape_check(
  const torch::Tensor& gradOutput,
  const torch::Tensor& indices,
  int64_t nbatch,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth,
  int nnumel, int64_t ndim, bool cuda=false)
{
  pool2d_shape_check(
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    nnumel, ndim);

//  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);

  // different CUDA/CPU behavior from TH
  if (cuda) {
    check_dim_size(indices, 4, 0, nbatch);
    check_dim_size(indices, 4, 1, nOutputPlane);
    check_dim_size(indices, 4, 2, outputHeight);
    check_dim_size(indices, 4, 3, outputWidth);
  }
  else {
    check_dim_size(indices, ndim, ndim-3, nOutputPlane);
    check_dim_size(indices, ndim, ndim-2, outputHeight);
    check_dim_size(indices, ndim, ndim-1, outputWidth);
  }
}

void max_pool2d_with_indices_out_cuda_template(
           torch::Tensor& output,
           torch::Tensor& indices,
           const torch::Tensor& input_,
           torch::IntArrayRef kernel_size,
           torch::IntArrayRef stride,
           torch::IntArrayRef padding,
           torch::IntArrayRef dilation,
           bool ceil_mode)
{
    torch::TensorArg output_arg{ output, "output", 1 };
    torch::TensorArg indices_arg{ indices, "indices", 2 };
    torch::TensorArg input_arg{ input_, "input_", 3 };

    torch::checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {output_arg, indices_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);

  pool2d_shape_check(
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    input_.numel(), input_.ndimension());

  torch::Tensor input = input_.contiguous(memory_format);
   

  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  output.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  indices.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  const int count = safe_downcast<int, int64_t>(output.numel());


      float *output_data = output.data_ptr<float>();
      float *input_data = input.data_ptr<float>();
      uint8_t *indices_data = indices.data_ptr<uint8_t>();

          max_pool_forward_nchw_cuda(
              count, input_data,
                  nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  output_data, indices_data);

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "max_pool2d_with_indices_out_cuda_frame failed with error code ",
     cudaGetLastError());

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

void max_pool2d_with_indices_backward_out_cuda_template(
           torch::Tensor& gradInput,
           const torch::Tensor& gradOutput_,
           const torch::Tensor& indices,
           torch::IntArrayRef input_size,
           torch::IntArrayRef input_stride,
           torch::IntArrayRef kernel_size,
           torch::IntArrayRef stride,
           torch::IntArrayRef padding,
           torch::IntArrayRef dilation,
           int64_t nnumel,
           int64_t input_ndimension,
           bool ceil_mode)
{
    torch::TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
    torch::TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
    torch::TensorArg indices_arg{ indices, "indices", 3 };

  torch::checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {gradInput_arg, gradOutput_arg, indices_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t nbatch = input_ndimension == 4 ? input_size[0] : 1;
  const int64_t nInputPlane = input_size[1];
  const int64_t inputHeight = input_size[2];
  const int64_t inputWidth = input_size[3];

  const int64_t in_stride_c = input_stride[0];
  const int64_t in_stride_h = input_stride[1];
  const int64_t in_stride_w = input_stride[2];

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    nnumel, input_ndimension,
    /*cuda=*/ true);

  const torch::Tensor gradOutput = gradOutput_.contiguous(at::MemoryFormat::Contiguous);

  const int64_t out_stride_c = gradOutput.stride(-3);
  const int64_t out_stride_h = gradOutput.stride(-2);
  const int64_t out_stride_w = gradOutput.stride(-1);

  gradInput.resize_(input_size);
  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(at::MemoryFormat::Contiguous);

  int64_t count = nnumel;

      float *gradOutput_data = gradOutput.data_ptr<float>();
      float *gradInput_data = gradInput.data_ptr<float>();
      uint8_t *indices_data = indices.data_ptr<uint8_t>();

          max_pool_backward_nchw_cuda(
              count,
                  gradOutput_data,
                  indices_data,
                  nbatch,
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  gradInput_data);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
    "fractional_max_pool2d_backward_out_cuda failed with error code ",
    cudaGetLastError());
}


std::tuple<torch::Tensor&, torch::Tensor&> max_pool2d_with_indices_out_cuda(
  torch::Tensor& output,
  torch::Tensor& indices,
  const torch::Tensor& input,
  torch::IntArrayRef kernel_size,
  torch::IntArrayRef stride,
  torch::IntArrayRef padding,
  torch::IntArrayRef dilation,
  bool ceil_mode)
{
  max_pool2d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<torch::Tensor&, torch::Tensor&>(output, indices);
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t> max_pool2d_with_indices_cuda(
  const torch::Tensor& input_,
  torch::IntArrayRef kernel_size,
  torch::IntArrayRef stride,
  torch::IntArrayRef padding,
  torch::IntArrayRef dilation,
  bool ceil_mode)
{

  torch::Tensor output = at::empty({0}, input_.options());
  torch::Tensor indices = at::empty({0}, input_.options().dtype(torch::kUInt8));
  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }
  const torch::Tensor input = input_.contiguous(memory_format);

  max_pool2d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);

  std::vector<int64_t> input_size = {input.ndimension() == 4 ? input.size(-4) : 1, input.size(-3), input.size(-2), input.size(-1)};
  std::vector<int64_t> input_stride = {input.stride(-3), input.stride(-2), input.stride(-1)};


  return std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t>(output, indices, input_size, input_stride, input.ndimension(), input.numel());
}

torch::Tensor& max_pool2d_with_indices_backward_out_cuda(
  torch::Tensor& gradInput,
  const torch::Tensor& gradOutput_,
  torch::IntArrayRef input_size,
  torch::IntArrayRef input_stride,
  const torch::Tensor& indices,
  torch::IntArrayRef kernel_size,
  torch::IntArrayRef stride,
  torch::IntArrayRef padding,
  torch::IntArrayRef dilation,
  int64_t input_ndimension,
  int64_t nnumel,
  bool ceil_mode)
{
  max_pool2d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    indices,
    input_size,
    input_stride,
    kernel_size,
    stride,
    padding,
    dilation,
    input_ndimension,
    nnumel,
    ceil_mode);
  return gradInput;
}

torch::Tensor& max_pool2d_with_indices_backward_out_cuda_normal(
  torch::Tensor& gradInput,
  const torch::Tensor& gradOutput_,
  const torch::Tensor& input,
  torch::IntArrayRef kernel_size,
  torch::IntArrayRef stride,
  torch::IntArrayRef padding,
  torch::IntArrayRef dilation,
  bool ceil_mode,
  const torch::Tensor& indices) {
    return at::max_pool2d_with_indices_backward_out(gradInput, gradOutput_, input, kernel_size,
              stride, padding, dilation, ceil_mode, indices);
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_with_indices_cuda", &max_pool2d_with_indices_cuda, "Sample non-zero values");
    m.def("max_pool2d_with_indices_backward_out_cuda", &max_pool2d_with_indices_backward_out_cuda, "Reconstruct vector from nonzero values and index bitmask");
    m.def("max_pool2d_with_indices_backward_out_cuda_normal", &max_pool2d_with_indices_backward_out_cuda_normal, "Reconstruct vector from nonzero values and index bitmask");
}