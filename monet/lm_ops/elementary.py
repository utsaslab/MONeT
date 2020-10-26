from .base import *
from .pack import *
import torch

#lrrelu_cpp = load(name="lrrelu_cpp", sources=[this_dir/"lrrelu.cpp"], extra_cflags=['-std=c++17'])
lravg_cpp = load(name="lravg_cpp", sources=[this_dir/"lravg.cpp"], extra_cflags=['-std=c++17'])

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


@implements(['aten::add', 'aten::add_'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
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
            if nodel:
                del stored
            s = self.params
            return grad_output, grad_output*s


@implements(['aten::flatten'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class Flatten(NativeOP):
    @staticmethod
    def op(x, start_dim, end_dim):
        return x.flatten(start_dim, end_dim)


@implements(['aten::t'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class T(NativeOP):
    @staticmethod
    def op(x):
        return x.t()


#@implements('aten::adaptive_avg_pool2d')
#class AdaptiveAvgPool2D(NativeOP):
#    @staticmethod
#    def op(x, os):
#        return torch.nn.functional.adaptive_avg_pool2d(x, os)

@implements(['aten::adaptive_avg_pool2d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
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
            return lravg_cpp.lr_adaptive_avg_pool_backward(grad_output,ip)


@implements(['aten::avg_pool2d'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class AvgPool2D(NativeOP):
    @staticmethod
    def op(*a):
        assert a[4] in [0, 1] and a[5] in [0, 1]
        a = list(a)
        a[4] = (a[4] == 1)
        a[5] = (a[5] == 1)
        return torch._C._nn.avg_pool2d(*a)

@implements(['aten::dropout'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class Dropout(NativeOP):
    @staticmethod
    def op(input_, p, training=False):
        return torch.nn.functional.dropout(input_, p, bool(training))