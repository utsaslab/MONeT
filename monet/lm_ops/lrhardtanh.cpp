#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

#include <math.h>
#include <stdlib.h>

torch::Tensor hardtanh_backward(torch::Tensor& grad, const torch::Tensor& self, float min, float max) {
    if (grad.type().is_cuda()) {
        return at::hardtanh_backward(grad, self, min, max);
    } else {
        exit(-1);
        // Not implementing CPU definition
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardtanh_backward", &hardtanh_backward, "HardTanh Backward");
}