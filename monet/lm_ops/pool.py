from .base import *
from .pack import *
import torch
import numpy as np

maxpool_cpp = load(name="maxpool_cpp", sources=[this_dir / "maxpool.cpp", this_dir / "maxpool.cu"], extra_cflags=['-std=c++17'])

@implements(['aten::max_pool2d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class SimpleMaxPool2D(OP):
    backward_storage = InputStorage(0)
    params = None

    # NOTE Because fwd calculated indices, the workspace memory will include the
    # memory of indice. Which means we are being slightly conservative in memory savings
    # for IndicesActMaxPool2D
    def forward(self, input_, kernel_size, stride, padding, dilation, ceil_mode):
        ceil_mode = bool(ceil_mode)
        self.params = kernel_size, stride, padding, dilation, ceil_mode
        with torch.no_grad():
            return torch.max_pool2d(input_, kernel_size, stride, padding, dilation, ceil_mode)

    def backward(self, grad_output, stored, nodel=False):
        ip = stored[0]
        if not nodel:
            del stored[0]
        kernel_size, stride, padding, dilation, ceil_mode = self.params
        with torch.enable_grad():
            tmp = torch.max_pool2d(ip.requires_grad_(True), kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, ceil_mode=ceil_mode)
        gradfn = tmp.grad_fn
        assert gradfn
        del tmp, ip
        with torch.no_grad():
            out = gradfn(grad_output)
            del gradfn
            return out

@implements(['aten::max_pool2d'], ['newnode', 'multiway_newnode', 'conv_multiway_newnode'])
class IndicesActMaxPool2D(OP):
    backward_storage = IntermediateStorage(lambda shape: (np.prod(shape)+1)//2) # Shape is output shape. Two indices are merged in 1 byte
    params = None

    def forward(self, input_, kernel_size, stride, padding, dilation, ceil_mode):
        with torch.no_grad():
            y, indices, input_size, input_stride, input_ndim, input_numel = maxpool_cpp.max_pool2d_with_indices_cuda(input_, 
                                                                            kernel_size, stride, padding, dilation, ceil_mode)
            index_shape = indices.shape
            index = indices.view(-1)
            del input_
            packed_indices = pack_two(index)
            del indices, index
            self.params = kernel_size, stride, padding, dilation, bool(ceil_mode), input_size, input_stride, input_ndim, input_numel, index_shape
            return y, packed_indices

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            packed_index = stored[0]
            if not nodel:
                del stored[0]
            kernel_size, stride, padding, dilation, ceil_mode, input_size, input_stride, input_ndim, input_numel, indices_shape = self.params
            index = unpack_two(packed_index)
            index.resize_(indices_shape)
            del packed_index
            gradInput = torch.zeros(input_size, device=grad_output.device)
            gradInput = maxpool_cpp.max_pool2d_with_indices_backward_out_cuda(gradInput, grad_output,
                                input_size, input_stride, index, kernel_size, stride, padding, 
                                dilation, input_numel, input_ndim, ceil_mode)
            del index
            return gradInput
