#include <cstdint>

__global__ void pack_two_kernel(const uint8_t * input, size_t N, uint8_t * r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2*j;
    uint8_t v = 0;
    if (i < N) {
        v |= (input[i] << 4);
        if (i + 1 < N) { v |= (input[i+1]);}
        r[j] = v;
    }
}
__global__ void unpack_two_kernel(const uint8_t * input, size_t N, uint8_t * r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2*j;
    if (i < N) {
        const uint8_t v = input[j];
        if (i+1<N) { r[i+1] = (v) & 15;}
        r[i] = (v >> 4) & 15;
    }
}
__global__ void pack_kernel(const bool * input, size_t N, uint8_t * r) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = 8*j;
    if (i < N) {
        uint8_t v = 0;
        for(int k=7; k >= 0 && i < N; i++, k--)
            v |= (input[i] << k);
        r[j] = v;
    }
}
__global__ void unpack_kernel(const uint8_t * input, size_t N, bool * r) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = 8*j;
    if (i < N) {
        const uint8_t v = input[j];
        for(int k=7; k >= 0 && i < N; i++, k--)
            r[i] = (v >> k) & 1;
    }
}
template<typename T>
__global__ void unpack_multiply_kernel(const uint8_t * input, const T * v, size_t N, T * r) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = 8*j;
    if (i < N) {
        const uint8_t t = input[j];
        for(int k=7; k >= 0 && i < N; i++, k--)
            r[i] = ((t >> k) & 1) * v[i];
    }
}

void pack_two_gpu(const uint8_t * input, size_t N, uint8_t * r) {
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    pack_two_kernel<<<blocks, threads>>>(input, N, r);
}
void unpack_two_gpu(const uint8_t * input, size_t N, uint8_t * r) {
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    unpack_two_kernel<<<blocks, threads>>>(input, N, r);
}
void pack_gpu(const bool * input, size_t N, uint8_t * r) {
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    pack_kernel<<<blocks, threads>>>(input, N, r);
}
void unpack_gpu(const uint8_t * input, size_t N, bool * r) {
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    unpack_kernel<<<blocks, threads>>>(input, N, r);
}
template<typename T>
void unpack_multiply_gpu(const uint8_t * input, const T * v, size_t N, T * r) {
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    unpack_multiply_kernel<<<blocks, threads>>>(input, v, N, r);
}
template void unpack_multiply_gpu<float>(const uint8_t *, const float *, size_t, float *);
template void unpack_multiply_gpu<double>(const uint8_t *, const double *, size_t, double *);
template void unpack_multiply_gpu<int>(const uint8_t *, const int *, size_t, int *);
template void unpack_multiply_gpu<uint8_t>(const uint8_t *, const uint8_t *, size_t, uint8_t *);
