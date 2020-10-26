#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

#include <math.h>


void threshold_bwd(float* , const float* , float , float , int);

torch::Tensor relu_backward(torch::Tensor& grad, const torch::Tensor& self, float threshold, int N) {
    float value = 0;
    if (grad.type().is_cuda()) {
        threshold_bwd(grad.data_ptr<float>(), self.data_ptr<float>(), threshold, value, N);
        return grad;
    } else {
        // Not implementing CPU definition
        return torch::zeros(N, torch::dtype(torch::kFloat32).device(grad.device()));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_backward", &relu_backward, "ReLU Backward");
}