import torch
from .base import *
from .pack import *
import numpy as np
from torch.utils.cpp_extension import load
from pathlib import Path

this_dir = Path(__file__).parent
myln_cpp = load(name="lrln_cpp", sources=[this_dir / "lrln.cpp", this_dir / "lrln.cu"], extra_cflags=['-std=c++17'])

# TODO - 1) recompute using precomputed statistics - done, 2) calculate output-activated backward

@implements(['aten::layer_norm'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal', 'gist'])
class InputActLayerNorm(OP):
    backward_storage = InputStorage(0, 2, 3)
    params = None   # params memory is unaccounted for => save_mean and save_var will occupy memory over the expected memory

    def forward(self, input_, shape, weight, bias,
                eps, use_cudnn=True, *args, **kwargs):
        with torch.no_grad():
            assert use_cudnn
            if self.params == None:  # if this is the first forward pass, save the values
                (out, mean, rstd), M, N = myln_cpp.forward(input_, shape, weight, bias, eps, use_cudnn)
                self.params = eps, mean, rstd, M, N, (input_.requires_grad, weight.requires_grad, bias.requires_grad)
            else:
                eps, mean, rstd, M, N, do_grad = self.params
                out = myln_cpp.forward_recompute(input_, shape, weight, bias, mean, rstd, M, N, eps)
            return out

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            input_, weight, bias = stored
            eps, mean, rstd, M, N, do_grad = self.params
            if not nodel:
                del stored[2], stored[1], stored[0]
            if grad_output.is_contiguous():
                di_app, dw_app, db_app = myln_cpp.cudnn_backward(grad_output, input_, mean, rstd, weight, M, N, do_grad)
            else:
                di_app, dw_app, db_app = myln_cpp.cudnn_backward(grad_output.contiguous(), input_, mean, rstd, weight, M, N, do_grad)
            return di_app, dw_app, db_app


# @implements(['aten::layer_norm'], ['multiway', 'multiway_newnode', 'conv_multiway_newnode'])
# class OutputActLayerNorm(OP):
#     backward_storage = [
#         InputStorage(2, 3),
#         OutputStorage(),
#     ]
#     params = None

#     def forward(self, input_, shape, weight, bias,
#                 eps, use_cudnn, *args, **kwargs):
#         with torch.no_grad():
#             assert use_cudnn
#             if self.params == None:  # if this is the first forward pass, save the values
#                 (out, mean, rstd), M, N = myln_cpp.forward(input_, shape, weight, bias, eps, use_cudnn)
#                 self.params = eps, mean, rstd, M, N, (input_.requires_grad, weight.requires_grad, bias.requires_grad)
#             else:
#                 # TODO - reuse stats
#                 eps, mean, rstd, M, N, do_grad = self.params
#                 (out, mean, rstd), M, N = myln_cpp.forward(input_, shape, weight, bias, eps, use_cudnn)
#             return out

#     def backward(self, grad_output, stored, nodel=False):
#         with torch.no_grad():
#             weight, bias, output = stored
#             eps, mean, rstd, M, N, do_grad = self.params
#             if not nodel:
#                 del stored[2], stored[1], stored[0]
#             # TODO
#             di_app, dw_app, db_app = myln_cpp.output_activate_lnorm_backward(grad_output, output, mean, rstd, weight, M, N, do_grad)
#             return di_app, dw_app, db_app