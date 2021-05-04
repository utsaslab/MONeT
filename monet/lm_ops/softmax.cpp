#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

using namespace at;
using namespace at::native;

torch::Tensor do_softmax_backward1(torch::Tensor &tmp, torch::Tensor &output, int64_t dim, bool half_to_float);

torch::Tensor softmax_forward(const torch::Tensor& self, int64_t dim) {
    torch::Tensor out = at::native::softmax_cuda(self, dim, false);
    namedinference::propagate_names(out, self);
    return out;
}

torch::Tensor softmax_backward1(const torch::Tensor& grad, torch::Tensor output, string input_type, int64_t dim) {
    ScalarType iptype;
    if (input_type == "torch.cuda.FloatTensor")
        iptype = ScalarType::Float;
    else if (input_type == "torch.cuda.HalfTensor")
        iptype = ScalarType::Half;
    else
        exit(-1);
    
    bool half_to_float = grad.scalar_type() != iptype;
    if (half_to_float) {
        TORCH_CHECK((grad.scalar_type() == ScalarType::Float && iptype == ScalarType::Half),
                    "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
    }
    torch::Tensor tmp = grad * output;
    return do_softmax_backward1(tmp, output, dim, half_to_float);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_forward, "Softmax forward");
    m.def("softmax_backward", &softmax_backward1, "Softmax backward");
}