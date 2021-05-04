from .base import *
from .pack import *
import torch

lrfuncs_cpp = load(name="lrfuncs_cpp", sources=[this_dir/"lrfuncs.cpp"], extra_cflags=['-std=c++17'])

@implements(['aten::adaptive_avg_pool2d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class AdaptiveAvgPool2D(OP):
    params = None

    def forward(self, x, os):
        with torch.no_grad():
            self.params = x.shape, x.device, x.requires_grad
            return torch._C._nn.adaptive_avg_pool2d(x, os)

    def backward(self, grad_output,stored, nodel=False):
        assert len(stored) == 0
        with torch.no_grad():
            shape, device, grad = self.params
            ip = torch.zeros(shape, device=device, requires_grad=grad)
            return lrfuncs_cpp.adaptive_avg_pool_backward(grad_output,ip)

@implements(['aten::add'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Add(OP):
    backward_storage = None
    params = None
    def forward(self, a, b, s):
        with torch.no_grad():
            self.params = s
            return torch.add(a, b, alpha=s)

    def backward(self, grad_output, stored, nodel=False):
        assert len(stored) == 0
        with torch.no_grad():
            if not nodel:
                del stored
            s = self.params
            return grad_output, grad_output*s

@implements(['aten::add_'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class AddIP(OP):
    backward_storage = None
    params = None
    inplace = False

    def forward(self, a, b, s):
        with torch.no_grad():
            self.params = s
            if self.inplace:
                return a.add_(b, alpha=s)
            return torch.add(a, b, alpha=s)

    def backward(self, grad_output, stored, nodel=False):
        assert len(stored) == 0
        with torch.no_grad():
            if not nodel:
                del stored
            s = self.params
            return grad_output, grad_output*s

@implements(['aten::embedding'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Embedding(OP):
    params = None
    backward_storage = InputStorage(1)

    def forward(self, weight, indices, padding_idx, scale_freq, is_sparse):
        with torch.no_grad():
            assert not is_sparse, "Will not handle sparse Embedding"
            out = lrfuncs_cpp.embedding(weight, indices, padding_idx, scale_freq, is_sparse)
            self.params = weight.shape[0], padding_idx, scale_freq#, is_sparse
            return out

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            indices = stores[0]
            if not nodel:
                del stores[0]
            num_weights, padding_idx, scale_freq = self.params
            return lrfuncs_cpp.embedding_backward(grad_output, indices, num_weights, padding_idx, scale_freq)

@implements(['aten::t_matmul'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class LowMemTMatMul(OP):
    params = None
    backward_storage = InputStorage(0,1)

    def forward(self, input1, input2):
        with torch.no_grad():
            input2_t = input2.t()
        out = torch.matmul(input1, input2_t)
        out_detached = out.detach()
        grad_name = out.grad_fn.next_functions[0][0].name()
        assert grad_name == "MmBackward"
        save_size2 = input2_t.size()
        save_stride2 = input2_t.stride()

        assert input1.dim() == 3 and input2.dim()==2
        input1_contiguous = input1.contiguous().view(-1, input1.shape[2])
        save_size1 = input1_contiguous.size()
        save_stride1 = input1_contiguous.stride()

        self.params = grad_name, save_size1, save_size2, save_stride1, save_stride2, (input1.requires_grad, input2_t.requires_grad)
        del out
        return out_detached

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            input1 = stores[0]
            input2 = stores[1]
            if not nodel:
                del stores[1]
                del stores[0]
            grad_name, size1, size2, stride1, stride2, do_grad = self.params
            dw, di = None, None
            if do_grad[1]:
                dw = lrfuncs_cpp.mm_mat2_backward(grad_output.view(-1,grad_output.shape[-1]), input1.view(size1), size2, stride2).t()
            del input1
            if do_grad[0]:
                di = lrfuncs_cpp.mm_mat1_backward(grad_output.view(-1,grad_output.shape[-1]), input2.t(), size1, stride1)
            return di, dw

@implements(['aten::matmul'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class LowMemMatMul(OP):
    params = None
    backward_storage = InputStorage(0,1)

    def forward(self, input1, input2):
        out = torch.matmul(input1, input2)
        out_detached = out.detach()
        grad_name = out.grad_fn.next_functions[0][0].name()

        save_size1 = input1.size()
        save_size2 = input2.size()
        save_stride1 = input1.stride()
        save_stride2 = input2.stride()

        if input1.dim() == 3 and input2.dim()==2:
            input1_contiguous = input1.contiguous().view(-1, input1.shape[2])
            save_size1 = input1_contiguous.size()
            save_stride1 = input1_contiguous.stride()
        elif input1.dim() == 4 and input2.dim() == 4:
            save_size1 = (input1.shape[0]*input1.shape[1],input1.shape[2], input1.shape[3])
            save_size2 = (input2.shape[0]*input2.shape[1],input2.shape[2], input2.shape[3])
        self.params = grad_name, save_size1, save_size2, save_stride1, save_stride2, (input1.requires_grad, input2.requires_grad)
        del out
        return out_detached

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            input1 = stores[0]
            input2 = stores[1]
            if not nodel:
                del stores[1]
                del stores[0]
            grad_name, size1, size2, stride1, stride2, do_grad = self.params
            dw, di = None, None
            if do_grad[1]:
                if grad_name == "MmBackward":
                    dw = lrfuncs_cpp.mm_mat2_backward(grad_output.view(-1,grad_output.shape[-1]), input1.view(size1), size2, stride2)
                elif grad_name == "BmmBackward":
                    dw = input1.view(size1).transpose(1, 2).bmm(grad_output.view(-1, grad_output.shape[2], grad_output.shape[3]))
                else:
                    raise RuntimeError("Not implemented %s" % grad_name)
            del input1
            if do_grad[0]:
                if grad_name == "MmBackward":
                    di = lrfuncs_cpp.mm_mat1_backward(grad_output.view(-1,grad_output.shape[-1]), input2.view(size2), size1, stride1)
                elif grad_name == "BmmBackward":
                    di = (grad_output.view(-1, grad_output.shape[2], grad_output.shape[3])).bmm(input2.view(size2).transpose(1, 2))
                else:
                    raise RuntimeError("Not implemented %s" % grad_name)
            return di, dw

@implements(['aten::slice'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Slice(OP):
    params = None
    backward_storage = None

    def forward(self, input_, dim, start, end, step):
        with torch.no_grad():
            self.params = input_.shape, dim, start, end, step
            return lrfuncs_cpp.slice(input_, dim, start, end, step)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            input_shape, dim, start, end, step = self.params
            return lrfuncs_cpp.slice_backward(grad_output, input_shape, dim, start, end, step)

@implements(['aten::select'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Select(OP):
    params = None
    backward_storage = None

    def forward(self, input_, dim, index):
        with torch.no_grad():
            self.params = input_.shape, dim, index
            return lrfuncs_cpp.select(input_, dim, index)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            input_shape, dim, index = self.params
            return lrfuncs_cpp.select_backward(grad_output, input_shape, dim, index)

@implements(['aten::tanh'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Tanh(OP):
    params = None
    backward_storage = OutputStorage()

    def forward(self, input_):
        with torch.no_grad():
            return torch.tanh(input_)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            output = stores[0]
            if not nodel:
                del stores[0]
            return lrfuncs_cpp.tanh_backward(grad_output, output)

@implements(['aten::gelu'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class GeLU(OP):
    params = None
    backward_storage = InputStorage(0)

    def forward(self, input_):
        with torch.no_grad():
            return torch.nn.functional.gelu(input_)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            input_ = stores[0]
            if not nodel:
                del stores[0]
            return lrfuncs_cpp.gelu_backward(grad_output, input_)

# rsub : output = other - input_ * alpha.
# Till now, we have observed other and alpha both to be Scalars.
# cannot be native OP, because grad_fn stores input_ unnecessarily
@implements(['aten::rsub'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class RSub(OP):
    params = None
    backward_storage = None

    def forward(self, input_, other, alpha):
        with torch.no_grad():
            assert not torch.is_tensor(other)
            self.params = alpha
            return lrfuncs_cpp.rsub_const(input_, other, alpha)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            alpha = self.params
            minus_alpha = -alpha
            return grad_output*minus_alpha

# @implements(['aten::rsub'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
# class RSub(OP):
#     params = None
#     backward_storage = None

#     def forward(self, input_, other, alpha):
#         with torch.no_grad():
#             assert not torch.is_tensor(other)
#             self.params = alpha
#             return lrfuncs_cpp.rsub_const(input_, other, alpha)

#     def backward(self, grad_output, stores, nodel=False):
#         with torch.no_grad():
#             alpha = self.params
#             minus_alpha = -alpha
#             return grad_output*minus_alpha

# @implements(['aten::zeros'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
# class Zeros(OP):
#     params = None
#     backward_storage = None

#     def forward(self, input_size, dtype, layout, device, pin):
#         with torch.no_grad():
#             import config
#             device = config.device
#             return lrfuncs_cpp.zeros(input_size, dtype, layout, device, pin)

#     def backward(self, grad_output, stores, nodel=False):
#         raise NotImplementedError

@implements(['aten::to'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class To(OP):
    params = None
    backward_storage = None

    def forward(self, input_, dtype, b1, b2, format):
        with torch.no_grad():
            assert format is None
            return lrfuncs_cpp.tofwd(input_, dtype, b1, b2)

    def backward(self, grad_output, stores, nodel=False):
        raise NotImplementedError

@implements(['aten::upsample_nearest3d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class UpsampleNearest3D(OP):
    params = None

    def forward(self, x, size1, size2, size3, scaled, scaleh, scalew):
        with torch.no_grad():
            size = [size1, size2, size3]
            out = torch._C._nn.upsample_nearest3d(x, size, scaled, scaleh, scalew)
            self.params = x.shape, size, scaled, scaleh, scalew
            return out

    def backward(self, grad_output, stored, nodel=False):
        assert len(stored) == 0
        with torch.no_grad():
            ipshape, size, scaled, scaleh, scalew = self.params
            return lrfuncs_cpp.lr_upsample_nearest_3d_backward(grad_output, size, ipshape, scaled, scaleh, scalew)