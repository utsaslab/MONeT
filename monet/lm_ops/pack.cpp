#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void pack_cpu(const bool * u, size_t N, uint8_t * p) {
    for(size_t i=0, j=0; i<N; j++) {
        uint8_t t = 0;
        for(int k=7; k >= 0 && i<N; i++, k--)
            t |= (u[i] << k);
        p[j] = t;
    }
}

void unpack_cpu(const uint8_t * p, size_t N, bool * u) {
    for(size_t i=0, j=0; i<N; j++) {
        uint8_t t = p[j];
        for(int k=7; k >= 0 && i<N; i++, k--)
            u[i] = (t >> k) & 1;
    }
}

template<typename T>
void unpack_multiply_cpu(const uint8_t * p, const T * v, size_t N, T * r) {
    for(size_t i=0, j=0; i<N; j++) {
        uint8_t t = p[j];
        for(int k=7; k >= 0 && i<N; i++, k--)
            r[i] = T((t >> k) & 1) * v[i];
    }
}
void pack_gpu(const bool *, size_t, uint8_t *);
void pack_two_gpu(const uint8_t *, size_t, uint8_t *);
void unpack_gpu(const uint8_t *, size_t, bool *);
void unpack_two_gpu(const uint8_t *, size_t, uint8_t *);
template<typename T>
void unpack_multiply_gpu(const uint8_t *, const T *, size_t N, T *);

torch::Tensor pack_two(const torch::Tensor& u) {
    TORCH_CHECK(u.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(u.scalar_type() == torch::kUInt8, "Only UInt8 tensors supported.");
    CHECK_CONTIGUOUS(u);
    size_t N = u.size(0);
    torch::Tensor p = torch::zeros((N+1) / 2, torch::dtype(torch::kUInt8).device(u.device()));
    if (u.type().is_cuda()) {
        pack_two_gpu(u.data_ptr<uint8_t>(), N, p.data_ptr<uint8_t>());
          TORCH_CHECK(cudaGetLastError() == cudaSuccess,
    "pack_two failed with error code ",
    cudaGetLastError());
    } else {
        std::cout<<"Unimplemented\n";
    }
    return p;
}
torch::Tensor unpack_two(const torch::Tensor& p, size_t N=0) {
    TORCH_CHECK(p.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(p.scalar_type() == torch::kUInt8, "Only uint8 tensors supported.");
    CHECK_CONTIGUOUS(p);
    if (!N)
        N = p.size(0) * 2;
    torch::Tensor u = torch::zeros(N, torch::dtype(torch::kUInt8).device(p.device()));
    if (p.type().is_cuda()) {
        unpack_two_gpu(p.data_ptr<uint8_t>(), N, u.data_ptr<uint8_t>());
          TORCH_CHECK(cudaGetLastError() == cudaSuccess,
    "unpack_two failed with error code ",
    cudaGetLastError());
    } else {
        std::cout<<"Unimplemented\n";
    }
    return u;
}

torch::Tensor pack(const torch::Tensor& u) {
    TORCH_CHECK(u.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(u.scalar_type() == torch::kBool, "Only bool tensors supported.");
    CHECK_CONTIGUOUS(u);
    size_t N = u.size(0);
    torch::Tensor p = torch::zeros((N-1) / 8 + 1, torch::dtype(torch::kUInt8).device(u.device()));
    if (u.type().is_cuda())
        pack_gpu(u.data_ptr<bool>(), N, p.data_ptr<uint8_t>());
    else
        pack_cpu(u.data_ptr<bool>(), N, p.data_ptr<uint8_t>());
    return p;
}
void pack_(const torch::Tensor& u, torch::Tensor& p) {
    TORCH_CHECK(u.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(u.scalar_type() == torch::kBool, "Only bool tensors supported.");
    CHECK_CONTIGUOUS(u);
    size_t N = u.size(0);
    TORCH_CHECK(p.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(p.scalar_type() == torch::kByte, "Only byte tensors supported.");
    TORCH_CHECK(p.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(p.size(0) == (N-1) / 8 + 1, "Only tensors size mismatch.");
    CHECK_CONTIGUOUS(p);
    TORCH_CHECK(u.type().is_cuda() == p.type().is_cuda(), "Device mismatch.");
    if (u.type().is_cuda())
        pack_gpu(u.data_ptr<bool>(), N, p.data_ptr<uint8_t>());
    else
        pack_cpu(u.data_ptr<bool>(), N, p.data_ptr<uint8_t>());
}
torch::Tensor unpack(const torch::Tensor& p, size_t N=0) {
    TORCH_CHECK(p.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(p.scalar_type() == torch::kUInt8, "Only uint8 tensors supported.");
    CHECK_CONTIGUOUS(p);
    if (!N)
        N = p.size(0) * 8;
    torch::Tensor u = torch::zeros(N, torch::dtype(torch::kBool).device(p.device()));
    if (p.type().is_cuda())
        unpack_gpu(p.data_ptr<uint8_t>(), N, u.data_ptr<bool>());
    else
        unpack_cpu(p.data_ptr<uint8_t>(), N, u.data_ptr<bool>());
    return u;
}
torch::Tensor unpack_multiply(const torch::Tensor& p, const torch::Tensor& v) {
    TORCH_CHECK(p.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(v.dim() == 1, "Only 1D tensors supported.");
    TORCH_CHECK(v.device() == p.device(), "All input need to be located on the same device.");
    TORCH_CHECK(p.scalar_type() == torch::kUInt8, "Only uint8 tensors supported.");
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(v);
    torch::Tensor r = torch::empty_like(v);
    size_t N = v.size(0);
    if (p.type().is_cuda())
        AT_DISPATCH_FLOATING_TYPES(v.type(), "unpack_multiply_gpu", ([&]{
            unpack_multiply_gpu<scalar_t>(p.data_ptr<uint8_t>(), v.data_ptr<scalar_t>(), N, r.data_ptr<scalar_t>());
        }));
    else
        AT_DISPATCH_FLOATING_TYPES(v.type(), "unpack_multiply_cpu", ([&]{
            unpack_multiply_cpu<scalar_t>(p.data_ptr<uint8_t>(), v.data_ptr<scalar_t>(), N, r.data_ptr<scalar_t>());
        }));
    return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_two", &pack_two, "Convert a uint8 tensor to a uint8 packed to half size");
    m.def("unpack_two", &unpack_two, "Convert a uint8 tensor to a double sized uint8 unpacked", py::arg("p"), py::arg("N") = 0);
    m.def("pack", &pack, "Convert a boolean tensor to a uint8 packed");
    m.def("pack_", &pack_, "Convert a boolean tensor to a uint8 packed");
    m.def("unpack", &unpack, "Convert a uint8 tensor to a boolean unpacked", py::arg("p"), py::arg("N") = 0);
    m.def("unpack_multiply", &unpack_multiply, "Convert a uint8 tensor to a boolean unpacked and multiply it with a second tensor", py::arg("p"), py::arg("v") = 0);
}
