#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& weight,
                      torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    static torch::Tensor undefined;
//    return torch::conv2d(input, weight, undefined, stride, padding, dilation, groups);
	return torch::cudnn_convolution(input, weight, undefined, padding, stride, dilation, groups, false, true); //benchmark, deterministic
}
torch::Tensor backward_input(torch::IntArrayRef input_sizes, const torch::Tensor& grad_output_t, const torch::Tensor& weight,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    // torch::Tensor grad_output = grad_output_t.contiguous(weight.suggest_memory_format());
    return torch::cudnn_convolution_backward_input(input_sizes, grad_output_t, weight, padding, stride, dilation, groups, false, true);
}

torch::Tensor backward_weight(torch::IntArrayRef weight_sizes,const torch::Tensor& grad_output_t, const torch::Tensor& input,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    // torch::NoGradGuard no_grad_guard;
    // torch::Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());
    return torch::cudnn_convolution_backward_weight(weight_sizes, grad_output_t, input, padding, stride, dilation, groups, false, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv forward");
    m.def("backward_input", &backward_input, "Conv backward input");
    m.def("backward_weight", &backward_weight, "Conv backward weight");
}
