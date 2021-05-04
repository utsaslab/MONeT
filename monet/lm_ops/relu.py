from .base import *
from .compress import *
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

@implements(['aten::relu'], ['gist'])
class IPActReLU(OP):
    backward_storage = InputStorage(0)
    params = None
    inplace = False

    def forward(self, x):
        with torch.no_grad():
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

@implements(['aten::relu_'], ['gist'])
class OutputActReLU(OP):
    backward_storage = OutputStorage()
    params = None
    inplace = True

    def forward(self, x):
        with torch.no_grad():
            return torch.relu_(x)

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

@implements(['aten::nosave_relu_'], ['gist'])
class NoSaveReLUIP(OP):
    backward_storage = OutputStorage()
    params = None
    inplace = True

    def forward(self, x):
        with torch.no_grad():
            out = torch.relu_(x)
            cip, col, row = compress_csr_256(out)
            intcol = col.to(torch.uint8)
            del col
            params = x.shape
            return out, (cip, intcol, row)

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            ip = stored[0]
            if not nodel:
                del stored[0]
            d = list(ip.shape)
            N = np.prod(d)
            y = lrrelu_cpp.relu_backward(x.view(N), ip.view(N), 0, N)
            return y.view(d)

@implements(['aten::nosave_relu'], ['gist'])
class NoSaveReLU(OP):
    backward_storage = InputStorage(0)
    params = None
    inplace = False

    def forward(self, x):
        with torch.no_grad():
            out = torch.relu(x)
            cip, col, row = compress_csr_256(out)
            intcol = col.to(torch.uint8)
            # params = x.shape
            return out, [cip, intcol, row]

    def backward(self, x, stored, nodel=False):
        with torch.no_grad():
            ip = stored[0]
            if not nodel:
                del stored[0]
            d = list(ip.shape)
            N = np.prod(d)
            y = lrrelu_cpp.relu_backward(x.view(N), ip.view(N), 0, N)
            return y.view(d)


@implements(['aten::savesign_relu'], ['gist'])
class GistBinaryActReLU(OP):
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

@implements(['aten::savesign_relu_'], ['gist'])
class GistBinaryActReLU(OP):
    backward_storage = IntermediateStorage(lambda shape: (np.prod(shape)+7)//8)
    params = None
    inplace = True

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