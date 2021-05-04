#include <torch/extension.h>
#include <torch/csrc/autograd/generated/Functions.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

using namespace at;
using namespace at::native;

torch::Tensor maybe_multiply1(const torch::Tensor & t, const Scalar & s) {
  bool is_one = false;
  if (s.isFloatingPoint()) {
    is_one = s.toDouble() == 1;
  } else if(s.isIntegral(true)) {
    is_one = s.toLong() == 1;
  }

  if (is_one) {
    return t;
  } else {
    return t * s;
  }
}

torch::Tensor adaptive_avg_pool_backward(const torch::Tensor& grad, const torch::Tensor& self) {
	return at::native::adaptive_avg_pool2d_backward_cuda(grad, self);
}

torch::Tensor embedding_forward(const torch::Tensor & weight, const torch::Tensor & indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

torch::Tensor embedding_backward1(const torch::Tensor & grad_, const torch::Tensor & indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
        return at::native::embedding_dense_backward_cuda(grad_, indices, num_weights, padding_idx, scale_grad_by_freq);
}

torch::Tensor mm_mat1_backward_t(const torch::Tensor & grad, const torch::Tensor & mat2, at::IntArrayRef mat1_sizes, at::IntArrayRef mat1_strides) {
    // Not using at call because it requires mat1 also (in pytorch v1.5.1)
    // return at::native::mm_mat1_backward(grad, mat2, mat1_sizes, mat1_strides, 1);
    // Assuming mat1 won't be sparse
    if (mat1_strides[0] == 1 && mat1_strides[1] == mat1_sizes[0]) {
        return maybe_multiply1((mat2.t()).mm(grad.t()).t(), 1);
    } else {
        return maybe_multiply1(grad.mm(mat2), 1);
    }
}

torch::Tensor mm_mat1_backward(const torch::Tensor & grad, const torch::Tensor & mat2, at::IntArrayRef mat1_sizes, at::IntArrayRef mat1_strides) {
    // Not using at call because it requires mat1 also (in pytorch v1.5.1)
    // return at::native::mm_mat1_backward(grad, mat2, mat1_sizes, mat1_strides, 1);
    // Assuming mat1 won't be sparse
    if (mat1_strides[0] == 1 && mat1_strides[1] == mat1_sizes[0]) {
        return maybe_multiply1(mat2.mm(grad.t()).t(), 1);
    } else {
        return maybe_multiply1(grad.mm(mat2.t()), 1);
    }
}

torch::Tensor mm_mat2_backward(const torch::Tensor & grad, const torch::Tensor & mat1, at::IntArrayRef sizes, at::IntArrayRef strides) {
    // return torch::autograd::generated::mm_mat2_backward(grad, mat1, sizes, strides, 1);
    if (strides[0] == 1 && strides[1] == sizes[0]) {
        if (mat1.is_sparse()) {
        // Since mm(dense, sparse) doesn't exist,
        // pass a transposed output matrix to the underlying "addmm"
        // function directly.
        int64_t out_rows = mat1.size(1);
        int64_t out_cols = grad.size(1);
        Tensor t = at::zeros({}, grad.options()).expand({out_rows, out_cols}, true);
        Tensor r = at::empty({out_cols, out_rows}, grad.options()).t();
        at::addmm_out(r, t, mat1.t(), grad, 1, 1);
        return r;
        }
        return maybe_multiply1(grad.t().mm(mat1).t(), 1);
    } else {
        return maybe_multiply1(mat1.t().mm(grad), 1);
    }
}

torch::Tensor slice_forward(const torch::Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
    return at::native::slice(self, dim, start, end, step);
}

torch::Tensor slice_backward1(const torch::Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
        // return at::native::slice_backward(grad, input_sizes, dim, start, end, step); // pytorch v1.7
        auto grad_input = at::zeros(input_sizes, grad.options());
        grad_input.slice(dim, start, end, step).copy_(grad);
        return grad_input;
}

torch::Tensor select_forward(const torch::Tensor& self, int64_t dim, int64_t index) {
    return at::native::select(self, dim, index);
}

torch::Tensor select_backward1(const torch::Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
        // return at::native::select_backward(grad, input_sizes, dim, index); // pytorch v1.7
        auto grad_input = at::zeros(input_sizes, grad.options());
        grad_input.select(dim, index).copy_(grad);
        return grad_input;
}

torch::Tensor tanh_backward1(const torch::Tensor& grad_output, const torch::Tensor& output) {
    return at::tanh_backward(grad_output, output);
}

torch::Tensor gelu_backward1(const torch::Tensor& grad_output, const torch::Tensor& input_) {
    return at::native::gelu_backward_cuda(grad_output, input_);
}

torch::Tensor rsub_const_forward(const torch::Tensor& input_, float other, int alpha) {
    return at::native::rsub(input_, at::Scalar(other), at::Scalar(alpha));
}

// torch::Tensor zeros(int size[], c10::ScalarType dtype, c10::Layout layout, c10::Device device, bool pin) {
//     return at::zeros(size, dtype, layout, device, pin);
// }

torch::Tensor tofwd(torch::Tensor input_, uint8_t dtype, bool b1, bool b2) {
    return at::native::to(input_, static_cast<c10::ScalarType>(dtype), b1, b2, c10::nullopt);
}

torch::Tensor lr_upsample_nearest_3d_backward(const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size,
    c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
	return at::native::upsample_nearest3d_backward_cuda(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_avg_pool_backward", &adaptive_avg_pool_backward, "Adaptive Avg Pool Backward");
    m.def("embedding", &embedding_forward, "Embedding forward");
    m.def("embedding_backward", &embedding_backward1, "Embedding backward");
    m.def("mm_mat1_backward", &mm_mat1_backward, "mm matrix1 backward");
    m.def("mm_mat1_backward_t", &mm_mat1_backward, "mm matrix1 backward whose saved input2 had been transposed before mm");
    m.def("mm_mat2_backward", &mm_mat2_backward, "mm matrix2 backward");
    m.def("slice", &slice_forward, "Slice forward");
    m.def("slice_backward", &slice_backward1, "Slice backward");
    m.def("select", &select_forward, "Select forward");
    m.def("select_backward", &select_backward1, "Select backward");
    // m.def("softmax", &softmax_forward, "Softmax forward");
    // m.def("softmax_backward", &softmax_backward1, "Softmax backward");
    m.def("tanh_backward", &tanh_backward1, "Tanh backward");
    m.def("gelu_backward", &gelu_backward1, "GeLU backward");
    m.def("rsub_const", &rsub_const_forward, "Rsub forward (other has type Scalar)");
    // m.def("zeros", &zeros, "Create zeros");
    m.def("tofwd", &tofwd, "To fwd impl");
    m.def("lr_upsample_nearest_3d_backward", &lr_upsample_nearest_3d_backward, "Upsample Nearest 3D Backward");
}