#include <torch/extension.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>

template <typename T>
__global__ void LayerNormForwardCUDAKernel1(
    int64_t N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC beta_v =
        beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
    Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
            static_cast<T_ACC>(rstd[i]) * gamma_v +
        beta_v;
  }
}

template <typename T>
void LayerNormForwardCUDA1(
    long M,
    long N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
        cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
        LayerNormForwardCUDAKernel1<T><<<M, 256, 0, cuda_stream>>>(
                N, X, mean, rstd, gamma, beta, Y);
    }

template void LayerNormForwardCUDA1(
    long M,
    long N,
    const double* X,
    const double* mean,
    const double* rstd,
    const double* gamma,
    const double* beta,
    double* Y);

template void LayerNormForwardCUDA1(
    long M,
    long N,
    const float* X,
    const float* mean,
    const float* rstd,
    const float* gamma,
    const float* beta,
    float* Y);

// void LayerNormForwardCUDA1(
//     int64_t M,
//     int64_t N,
//     const c10::Half* X,
//     const c10::Half* mean,
//     const c10::Half* rstd,
//     const c10::Half* gamma,
//     const c10::Half* beta,
//     c10::Half* Y);