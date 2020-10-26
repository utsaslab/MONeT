#include <cstdint>
#include <stdio.h>

__global__ void threshold_kernel(float* grad, const float* self, float threshold, float value, int N){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j<N) {
        grad[j] = self[j] > threshold ? grad[j] : value;
    }
}

void threshold_bwd(float* grad, const float* self, float threshold, float value, int N) {
    const int threads = 1024;
    const int blocks = (N+threads-1)/threads;
    threshold_kernel<<<blocks, threads>>>(grad, self, threshold, value, N);
}
