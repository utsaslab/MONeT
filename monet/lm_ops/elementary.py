from .base import *
from .pack import *
import torch

class NativeOP(OP):
    op = None
    grad_fn = None
    backward_storage = None

    def forward(self, input_, *args, **kwargs):
        with torch.enable_grad():
            r = self.op(input_.requires_grad_(True), *args, **kwargs)
        assert r.grad_fn is not None
        self.grad_fn = r.grad_fn
        rcopy = r.detach()
        del r, input_
        for arg in args:
            del arg
        return rcopy

    def backward(self, grad_output, stored, nodel=False):
        assert len(stored) == 0
        return self.grad_fn(grad_output)


@implements(['aten::flatten'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Flatten(NativeOP):
    @staticmethod
    def op(x, start_dim, end_dim):
        return x.flatten(start_dim, end_dim)

@implements(['aten::permute'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Permute(NativeOP):
    @staticmethod
    def op(x, order):
        return x.permute(order)

@implements(['aten::transpose'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Transpose(NativeOP):
    @staticmethod
    def op(x, dim0, dim1):
        return torch.transpose(x,dim0,dim1)

@implements(['aten::view'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class View(NativeOP):
    @staticmethod
    def op(x, dim_list):
        return x.view(dim_list)

@implements(['aten::div'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Div(NativeOP):
    @staticmethod
    def op(x, divisor):
        return torch.div(x, divisor)

@implements(['aten::contiguous'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Contiguous(NativeOP):
    @staticmethod
    def op(x, memory_format):
        assert memory_format == 0
        return x.contiguous()

@implements(['aten::unsqueeze'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Unsqueeze(OP):
    def forward(self, x, dim):
        assert not x.requires_grad
        return torch.unsqueeze(x, dim)

@implements(['aten::mul'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Mul(NativeOP):
    @staticmethod
    def op(x, alpha):
        return x * alpha

@implements(['aten::t'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class T(NativeOP):
    @staticmethod
    def op(x):
        return x.t()

@implements(['aten::avg_pool2d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class AvgPool2D(NativeOP):
    @staticmethod
    def op(*a):
        assert a[4] in [0, 1] and a[5] in [0, 1]
        a = list(a)
        a[4] = (a[4] == 1)
        a[5] = (a[5] == 1)
        return torch._C._nn.avg_pool2d(*a)

@implements(['aten::dropout'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class Dropout(NativeOP):
    @staticmethod
    def op(input_, p, training=False):
        return torch.nn.functional.dropout(input_, p, bool(training))