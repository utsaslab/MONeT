#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit
#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int origp_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

// kernels borrowed from Caffe
__global__ void max_pool_forward_nchw(const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, float* top_data,
    uint8_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    uint8_t count_h = 0;
    uint8_t count_w = 0;
    while(hstart < 0) {
      hstart += dilation_h;
      count_h++;
    }
    while(wstart < 0) {
      wstart += dilation_w;
      count_w++;
    }
    uint8_t ch = count_h;
    uint8_t cw = count_w; 
    float maxval = at::numeric_limits<float>::lower_bound(); // -Infinity
    int maxidx = hstart * width + wstart;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        float val = bottom_data[h * width + w];
        if ((ScalarConvert<float, float>::to(val) > maxval) || THCNumerics<float>::isnan(val)) {
          maxidx = ch*(uint8_t)kernel_w + cw;
          maxval = ScalarConvert<float, float>::to(val);
        }
        cw += 1;
      }
      cw = count_w;
      ch += 1;
    }
    top_data[index] = ScalarConvert<float, float>::to(maxval);
    top_mask[index] = maxidx;
  }
}

static const int BLOCK_THREADS = 256;

#if defined (__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 4)
#else
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 8)
#endif
__global__ void max_pool_backward_nchw(const int nthreads, const float* top_diff,
    const uint8_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    float* bottom_diff) {
//    printf("Thread: %d, %d, %d, Block: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z); 
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (height*width); index += blockDim.x * gridDim.x) {
    int h = index/width;
    int w = index - h * width;
    int phstart = p_start(h, pad_h, kernel_h, dilation_h, stride_h);
    int origphstart = origp_start(h, pad_h, kernel_h, dilation_h, stride_h);
    int phend = p_end(h, pad_h, pooled_height, stride_h);
    int pwstart = p_start(w, pad_w, kernel_w, dilation_w, stride_w);
    int origpwstart = origp_start(w, pad_w, kernel_w, dilation_w, stride_w);
    int pwend = p_end(w, pad_w, pooled_width, stride_w);
    for (int n = blockIdx.y; n < num; n += gridDim.y)
       for (int c = blockIdx.z; c < channels; c+= gridDim.z) {

        float gradient = 0;
        int offset = (n * channels + c) * pooled_height * pooled_width;
//        top_diff += offset;
//        top_mask += offset;
        uint8_t this_element_idx;

        if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
              // This works for very few cases
              this_element_idx = (uint8_t)(((h-ph*stride_h+pad_h)/dilation_h)*kernel_w + (w-pw*stride_w+pad_w)/dilation_w);
//              printf("%d, %d, %d, %d, %d, %d \n", h,w,this_element_idx,ph,pw,ph * pooled_width + pw);
            if (top_mask[offset + ph * pooled_width + pw] == this_element_idx) {
//                printf("%d, %d, %d, %d, %d, %d, %d\n", h,w,ph,pw, (int32_t)this_element_idx, top_mask[ph * pooled_width + pw], ph * pooled_width + pw);
              gradient += ScalarConvert<float, float>::to(top_diff[offset + ph * pooled_width + pw]);
            }
          }
        }
        } else {
            this_element_idx = (uint8_t)(((h - phstart * stride_h + pad_h)/dilation_h)*kernel_w + (w - pwstart*stride_w + pad_w)/dilation_w);
//              printf("%d, %d, %d, %d, %d, %d \n", h,w,this_element_idx,phstart,pwstart,phstart * pooled_width + pwstart);
            if (top_mask[offset + phstart * pooled_width + pwstart] == this_element_idx) {
//                printf("%d, %d, %d, %d, %d, %d, %d\n", h,w,phstart,pwstart, (int32_t)this_element_idx, top_mask[phstart * pooled_width + pwstart], phstart * pooled_width + pwstart);
              gradient += ScalarConvert<float, float>::to(top_diff[offset+phstart * pooled_width + pwstart]);
            }
        }
        //printf("%d\n",(n*channels+c)*height*width+index);
        bottom_diff[(n*channels+c)*height*width+index] = ScalarConvert<float, float>::to(gradient);
      }
  }
}

#define MAX_THREADS_PER_BLOCK 1024

void max_pool_forward_nchw_cuda(const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, float* top_data,
    uint8_t* top_mask) {
          const int num_threads = std::min(MAX_THREADS_PER_BLOCK,
                                           BLOCK_THREADS);
//          std::cout<<nthreads<<" "<<num<<" "<<channels<<" "<<height<<" "<<width<<" "<<pooled_height<<" "<<pooled_width<<" "<<kernel_h<<" "<<kernel_w<<" "<<stride_h<<" "<<stride_w<<" "<<pad_h<<" "<<pad_w<<" "<<dilation_h<<" "<<dilation_w<<" "<<num_threads<<"\n";
          max_pool_forward_nchw
              <<<(nthreads + num_threads - 1)/num_threads, num_threads>>>(
        nthreads, bottom_data,
        num, channels, height,
        width, pooled_height, pooled_width,
        kernel_h, kernel_w, stride_h,
        stride_w, pad_h, pad_w,
        dilation_h, dilation_w, top_data,
        top_mask);
}

void max_pool_backward_nchw_cuda(const int nthreads, const float* top_diff,
    const uint8_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    float* bottom_diff) {
          int imgcount = width * height;
          dim3 grid;
          const int blocks = (imgcount + BLOCK_THREADS - 1) / BLOCK_THREADS;
          grid.x = blocks;
          grid.y = num;
          uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
          if (maxGridY < grid.y) grid.y = maxGridY;
          grid.z = channels;
          uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
          if (maxGridZ < grid.z) grid.z = maxGridZ;

//           printf("grid.x %lld, grid.y %lld, grid.z %lld, maxY %lld, maxZ %lld\n \
//                   nthreads %lld, num %lld, blocks %d\n \
//                  c  %lld, h %lld, w %lld, ph %lld, pw %lld,\n \
//                  kh %lld, kw %lld, sh %lld, sw %lld, ph %lld, pw %lld, dh %lld, dw %lld\n", grid.x, grid.y, grid.z, maxGridY, maxGridZ,
//                  nthreads, num, blocks,
//                  channels, height, width, pooled_height, pooled_width,
//                  kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,dilation_h, dilation_w);
          max_pool_backward_nchw
          <<<grid, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              nthreads,
                  top_diff,
                  top_mask,
                  num,
                  channels, height, width, pooled_height, pooled_width,
                  kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, bottom_diff);
          }