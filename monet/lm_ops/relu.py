from .base import *
from .pack import *
import torch
import numpy as np

lrrelu_cpp = load(name="lrrelu_cpp", sources=[this_dir/"lrrelu.cpp", this_dir/"lrrelu.cu"], extra_cflags=['-std=c++17'])

@implements(['aten::relu', 'aten::relu_'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode'])
class InputActReLU(OP):
    backward_storage = InputStorage(0)
    params = None
    inplace = False

    def forward(self, x):
        with torch.no_grad():
            if self.inplace:
                return torch.relu_(x)
            return torch.relu(x)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            ip = stored[0]
            if not nodel:
                del stored[0]
            d = list(ip.shape)
            N = np.prod(d)
            y = lrrelu_cpp.relu_backward(x.view(N), ip.view(N), 0, N)
            return y.view(d)


@implements(['aten::relu', 'aten::relu_'], ['multiway', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class OutputActReLU(OP):
    backward_storage = OutputStorage()
    params = None
    inplace = False

    def forward(self, x):
        with torch.no_grad():
            if self.inplace:
                return torch.relu_(x)
            return torch.relu(x)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            op = stored[0]
            if not nodel:
                del stored[0]
            d = list(op.shape)
            N = np.prod(d)
            y = lrrelu_cpp.relu_backward(x.view(N), op.view(N), 0, N)
            return y.view(d)

@implements(['aten::relu', 'aten::relu_'], ['newnode', 'multiway_newnode', 'conv_multiway_newnode'])
class BinaryActReLU(OP):
    backward_storage = IntermediateStorage(lambda shape: (np.prod(shape)+7)//8)
    params = None
    inplace = False

    def forward(self, x):
        with torch.no_grad():
            if self.inplace:
                return x.clamp_min_(0), pack((x > 0).view(-1))
            return x.clamp_min(0), pack((x > 0).view(-1))

    def intermediate(self, x):
        return pack((x > 0).view(-1))

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            sign_pack = stored[0]
            if not nodel:
                del stored[0]
            sign = unpack(sign_pack)
            del sign_pack
            shape = x.shape
            x *= sign.view(shape)
            return x
            # Above is two times faster
            # return unpack_multiply(ip, x.view(-1)).view(*x.shape)
