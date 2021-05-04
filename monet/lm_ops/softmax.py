from .base import *
from .pack import *
import torch

softmax_cpp = load(name="softmax_cpp", sources=[this_dir/"softmax.cpp", this_dir/"softmax.cu"], extra_cflags=['-std=c++17'])

@implements(['aten::softmax'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class Softmax(OP):
    params = None
    backward_storage = OutputStorage()

    def forward(self, input_, dim, dtype):
        with torch.no_grad():
            assert dtype is None
            self.params = input_.type(), dim
            return softmax_cpp.softmax(input_, dim)

    def backward(self, grad_output, stores, nodel=False):
        with torch.no_grad():
            output = stores[0]
            if not nodel:
                del stores[0]
            input_type, dim = self.params
            return softmax_cpp.softmax_backward(grad_output, output, input_type, dim)