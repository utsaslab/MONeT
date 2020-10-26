#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


torch::Tensor lr_adaptive_avg_pool_backward(const torch::Tensor& grad, const torch::Tensor& self) {
	return at::native::adaptive_avg_pool2d_backward_cuda(grad, self);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lr_adaptive_avg_pool_backward", &lr_adaptive_avg_pool_backward, "Adaptive Avg Pool Backward");
}
