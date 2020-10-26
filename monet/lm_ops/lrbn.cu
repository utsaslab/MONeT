#include <torch/extension.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>
// #include "lrbn.cuh"

using namespace at;
using namespace at::native;

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, C10_WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of C10_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most C10_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than C10_WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[C10_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % C10_WARP_SIZE == 0) {
    shared[tid / C10_WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / C10_WARP_SIZE && tid < C10_WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / C10_WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel1(
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
    bool train) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  stat_accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = stat_accscalar_t(0);
    invstd = stat_accscalar_t(1);
  }

  stat_accscalar_t weight_val = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  stat_accscalar_t bias_val = bias.size(0) > 0 ? static_cast<stat_accscalar_t>(bias[plane]) : stat_accscalar_t(0);
  stat_accscalar_t norm = stat_accscalar_t(1) / N;
  stat_accscalar_t inv_weight = stat_accscalar_t(1) / weight_val;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<input_scalar_t, stat_accscalar_t, torch::GenericPackedTensorAccessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t>> g(bias_val, input, grad_output);
  Float2<input_scalar_t, stat_accscalar_t> res = reduce<Float2<input_scalar_t, stat_accscalar_t>, GradOp<input_scalar_t, stat_accscalar_t,
                                                                                   torch::GenericPackedTensorAccessor<input_scalar_t, 3, torch::DefaultPtrTraits, index_t>>>(g, grad_output, plane);
  stat_accscalar_t grad_output_sum = res.v1;
  stat_accscalar_t dot_p = res.v2;

  stat_accscalar_t grad_mean = grad_output_sum * norm;
  stat_accscalar_t proj_scale = dot_p * norm * inv_weight * inv_weight;
  stat_accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        input_scalar_t go = grad_output[batch][plane][x];
        if (train) {
          stat_accscalar_t inp = input[batch][plane][x];
          stat_accscalar_t proj = (inp - bias_val) * proj_scale;
          grad_input[batch][plane][x] = static_cast<input_scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<input_scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * inv_weight);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
    }
  }
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
    bool train) {
      using accscalar_t = at::acc_type<stat_scalar_t, true>;

      auto stream = at::cuda::getCurrentCUDAStream();
      dim3 blocks(input.size(1));
      int tf = getNumThreads(input.size(2));
      dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

      batch_norm_backward_kernel1<input_scalar_t, stat_scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
      (input, grad_output, grad_input, grad_weight, grad_bias, weight, bias, running_mean, running_var,
       save_mean, save_invstd, train);
}

template void batch_norm_bwd1<double, double, double, int>(
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> input,
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> grad_output,
  torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> grad_input,
  torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> grad_weight,
  torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> grad_bias,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> weight,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> bias,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> running_mean,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> running_var,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> save_mean,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, int> save_invstd,
  bool train);

template void batch_norm_bwd1<double, double, double, long>(
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> input,
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> grad_output,
  torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> grad_input,
  torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> grad_weight,
  torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> grad_bias,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> weight,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> bias,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> running_mean,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> running_var,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> save_mean,
  const torch::GenericPackedTensorAccessor<double, 1, torch::DefaultPtrTraits, long> save_invstd,
  bool train);

template void batch_norm_bwd1<double, float, float, int>(
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> input,
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> grad_output,
  torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, int> grad_input,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> grad_weight,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> grad_bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> weight,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> running_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> running_var,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> save_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> save_invstd,
  bool train);

template void batch_norm_bwd1<double, float, float, long>(
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> input,
  const torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> grad_output,
  torch::GenericPackedTensorAccessor<double, 3, torch::DefaultPtrTraits, long> grad_input,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> grad_weight,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> grad_bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> weight,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> running_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> running_var,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> save_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> save_invstd,
  bool train);

template void batch_norm_bwd1<float, float, float, int>(
  const torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, int> input,
  const torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, int> grad_output,
  torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, int> grad_input,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> grad_weight,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> grad_bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> weight,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> running_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> running_var,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> save_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, int> save_invstd,
  bool train);

template void batch_norm_bwd1<float, float, float, long>(
  const torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, long> input,
  const torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, long> grad_output,
  torch::GenericPackedTensorAccessor<float, 3, torch::DefaultPtrTraits, long> grad_input,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> grad_weight,
  torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> grad_bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> weight,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> bias,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> running_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> running_var,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> save_mean,
  const torch::GenericPackedTensorAccessor<float, 1, torch::DefaultPtrTraits, long> save_invstd,
  bool train);
