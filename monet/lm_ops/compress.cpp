#include <torch/extension.h>
#include <chrono>
#include <ctime>
#include<cstdio>
#include<cstdlib>
#include<cstdint>
#include <cusparse.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void compress_csr_256_gpu(const float* ip, float* cip, int* idx, int * rowidx, const int * nnzPerRow, size_t N);
void uncompress_csr_256_gpu(const float* compIP, const int * csrIdx, const int* csrRowIdx, float* op, size_t N);
// void get_nz_cuda(const float* ip, int *nnzPerRow, int nz_ptr, int lda);

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> compress_csr_256(const torch::Tensor& ip, const torch::Tensor& nnzPerRow, size_t nz) {
    TORCH_CHECK(ip.dim() == 2 && ip.size(1) == 256, "Only N/256 x 256 tensors supported.");
    CHECK_CONTIGUOUS(ip);
    // size_t N = ip.size(0) * ip.size(1);
    // int lda = (N+255)/256;
    // int nz1;
    // int *nnzPerRow1;
    // get_nz_cuda(ip.data_ptr<float>(), nnzPerRow1, nz1, lda);
    // std::cout << "nz: "<< nz << std::endl;
    TORCH_CHECK(nnzPerRow.scalar_type() == torch::kInt32, "nnzPerRow should be int32.");
    size_t N = ip.size(0) * ip.size(1);
    torch::Tensor cip = torch::zeros(nz, torch::dtype(torch::kFloat32).device(ip.device()));
    torch::Tensor idx = torch::zeros(nz, torch::dtype(torch::kInt32).device(ip.device()));
    torch::Tensor rowidx = torch::zeros(ip.size(0)+1, torch::dtype(torch::kInt32).device(ip.device()));
    if (ip.type().is_cuda()) {
        // compress_csr_256_gpu(ip.data_ptr<float>(), cip.data_ptr<float>(), idx.data_ptr<int>(), rowidx.data_ptr<int>(), nnzPerRow.data_ptr<int>(), N, nz);
        compress_csr_256_gpu(ip.data_ptr<float>(), cip.data_ptr<float>(), idx.data_ptr<int>(), rowidx.data_ptr<int>(), nnzPerRow.data_ptr<int>(), N);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
            "compress_csr_256 failed with error code ",
            cudaGetLastError());
    } else {
        std::cout<<"CPU type\n";
    }
    return {cip, idx, rowidx};
}

torch::Tensor uncompress_csr_256(const torch::Tensor& compip, const torch::Tensor& indx, const torch::Tensor& rowidx, size_t N) {
    TORCH_CHECK(compip.dim() == 1 , "Only 1D tensors supported.");
    CHECK_CONTIGUOUS(compip);
    size_t nz = compip.size(0);
    torch::Tensor op = torch::zeros({(N+255)/256,256}, torch::dtype(torch::kFloat32).device(compip.device()));
    torch::Tensor nnzPerRow = torch::full({1}, (int)nz, torch::dtype(torch::kInt32).device(compip.device()));

    if (compip.type().is_cuda()) {
        uncompress_csr_256_gpu(compip.data_ptr<float>(), indx.data_ptr<int>(), rowidx.data_ptr<int>(), op.data_ptr<float>(), N);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
            "uncompress_csr_256 failed with error code ",
            cudaGetLastError());
    } else {
        std::cout<<"CPU type\n";
    }
    return op;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_csr_256", &compress_csr_256, "Convert values to a csr form");
    m.def("uncompress_csr_256", &uncompress_csr_256, "reconstruct vector from nonzero values and nonzero indices");
}

