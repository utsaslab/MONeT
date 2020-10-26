from .base import *
from .pack import *
import torch
import numpy as np

lrhardtanh_cpp = load(name="lrhardtanh_cpp", sources=[this_dir/"lrhardtanh.cpp"], extra_cflags=['-std=c++17'])


@implements(['aten::hardtanh'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class InputActHardTanh(OP):
    backward_storage = InputStorage(0)
    params = None
    inplace = False

    def forward(self, x, lower, upper):
        with torch.no_grad():
            self.params = lower, upper
            if self.inplace:
                return torch.nn.functional.hardtanh_(x, lower, upper)
            return torch.nn.functional.hardtanh(x, lower, upper)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            lower, upper = self.params
            ip = stored[0]
            if not nodel:
                del stored[0]
            return lrhardtanh_cpp.hardtanh_backward(x, ip, lower, upper)


@implements(['aten::hardtanh'], ['multiway', 'multiway_newnode', 'conv_multiway_newnode'])
class OutputActHardTanh(OP):
    backward_storage = OutputStorage()
    params = None
    inplace = False

    def forward(self, x, lower, upper):
        with torch.no_grad():
            self.params = lower, upper
            if self.inplace:
                return torch.nn.functional.hardtanh_(x, lower, upper)
            return torch.nn.functional.hardtanh(x, lower, upper)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            op = stored[0]
            lower, upper = self.params
            if not nodel:
                del stored[0]
            return lrhardtanh_cpp.hardtanh_backward(x, op, lower, upper)

@implements(['aten::hardtanh_'], ['multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode'])
class InputActHardTanh(OP):
    backward_storage = InputStorage(0)
    params = None
    inplace = False

    def forward(self, x, lower, upper):
        with torch.no_grad():
            self.params = lower, upper
            if self.inplace:
                return torch.nn.functional.hardtanh_(x, lower, upper)
            return torch.nn.functional.hardtanh(x, lower, upper)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            lower, upper = self.params
            ip = stored[0]
            if not nodel:
                del stored[0]
            return lrhardtanh_cpp.hardtanh_backward(x, ip, lower, upper)


@implements(['aten::hardtanh_'], ['normal','multiway', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class OutputActHardTanh(OP):
    backward_storage = OutputStorage()
    params = None
    inplace = False

    def forward(self, x, lower, upper):
        with torch.no_grad():
            self.params = lower, upper
            if self.inplace:
                return torch.nn.functional.hardtanh_(x, lower, upper)
            return torch.nn.functional.hardtanh(x, lower, upper)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            op = stored[0]
            lower, upper = self.params
            if not nodel:
                del stored[0]
            return lrhardtanh_cpp.hardtanh_backward(x, op, lower, upper)