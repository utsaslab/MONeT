from .base import *
import torch


@implements(['aten::cat'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class Cat(OP):
    params = None

    def forward(self, *input_):
        tensors, dim = input_[:-1], input_[-1]
        self.params = dim, [x.shape[dim] for x in tensors]

        with torch.no_grad():
            if len(input_) == 2:
                return input_[0]
            return torch.cat(tensors, dim=dim)

    def backward(self, *grad_output, nodel=False):
        dim, chunk_sizes = self.params
        with torch.no_grad():
            if len(chunk_sizes) == 1:
                return grad_output[0]
            outs = list(torch.split(
                grad_output[0], chunk_sizes, dim=dim))
            grad_inputs = []
            for out in outs:
                grad_inputs.append(out.contiguous())
            del outs, grad_output
            return grad_inputs

