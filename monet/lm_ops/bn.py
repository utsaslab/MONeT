import torch
from .base import *
from .pack import *
import numpy as np
from torch.utils.cpp_extension import load
from pathlib import Path

this_dir = Path(__file__).parent
mybn_cpp = load(name="lrbn_cpp", sources=[this_dir / "lrbn.cpp", this_dir / "lrbn.cu"], extra_cflags=['-std=c++17'])


@implements(['aten::batch_norm'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class InputActBatchNorm(OP):
    backward_storage = InputStorage(0, 1, 2, 3, 4)
    params = None   # params memory is unaccounted for => save_mean and save_var will occupy memory over the expected memory
    running_mean = None
    running_var = None

    def forward(self, input_, weight, bias,
                running_mean, running_var,
                training=0, momentum=0.1, eps=1e-5, *args, **kwargs):
        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = running_mean.requires_grad_(False)
                self.running_var = running_var.requires_grad_(False)

            if self.params == None:  # if this is the first forward pass, save the values
                out, save_mean, save_var, res_space = mybn_cpp.forward(
                    input_, self.running_mean, self.running_var, weight, bias,
                    training, momentum, eps)
                self.params = training, eps, save_mean, save_var, res_space
            else:
                training, eps, batch_mean, batch_var, res_space = self.params
                out = torch.nn.functional.batch_norm(input_, batch_mean, 1/torch.square(batch_var), weight, bias, False, momentum, eps)
            return out

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            input_, weight, bias, running_mean, running_var = stored
            training, eps, save_mean, save_var, res_space = self.params

            if training:
                di_app, dw_app, db_app = mybn_cpp.cudnn_backward(
                    input_, grad_output, weight,
                    running_mean, running_var,
                    save_mean, save_var,
                    eps, res_space)
            else:
                raise NotImplementedError

            if not nodel:
                del stored[4], stored[3], stored[2], stored[1], stored[0]

            return di_app, dw_app, db_app, None, None

@implements(['aten::batch_norm'], ['gist'])
class InputActBatchNorm(OP):
    backward_storage = InputStorage(0, 1, 2, 3, 4)
    params = None   # params memory is unaccounted for => save_mean and save_var will occupy memory over the expected memory
    running_mean = None
    running_var = None

    def forward(self, input_, weight, bias,
                running_mean, running_var,
                training=0, momentum=0.1, eps=1e-5, *args, **kwargs):
        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = running_mean.requires_grad_(False)
                self.running_var = running_var.requires_grad_(False)

            out, save_mean, save_var, res_space = mybn_cpp.forward(
                input_, self.running_mean, self.running_var, weight, bias,
                training, momentum, eps)
            self.params = training, eps, save_mean, save_var, res_space
            return out

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            input_, weight, bias, running_mean, running_var = stored
            training, eps, save_mean, save_var, res_space = self.params

            if training:
                di_app, dw_app, db_app = mybn_cpp.cudnn_backward(
                    input_, grad_output, weight,
                    running_mean, running_var,
                    save_mean, save_var,
                    eps, res_space)
            else:
                raise NotImplementedError

            if not nodel:
                del stored[4], stored[3], stored[2], stored[1], stored[0]

            return di_app, dw_app, db_app, None, None

@implements(['aten::batch_norm'], ['multiway', 'multiway_newnode', 'conv_multiway_newnode'])
class OutputActBatchNorm(OP):
    backward_storage = [
        InputStorage(1, 2, 3, 4),
        OutputStorage(),
    ]
    params = None
    running_mean = None
    running_var = None

    def forward(self, input_, weight, bias,
                running_mean, running_var,
                training=False, momentum=0.1, eps=1e-5, *args, **kwargs):
        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = running_mean.requires_grad_(False)
                self.running_var = running_var.requires_grad_(False)

            if self.params == None:  # if this is the first forward pass, save the values
                out, save_mean, save_var, res_space = mybn_cpp.forward(
                    input_, self.running_mean, self.running_var, weight, bias,
                    training, momentum, eps)
                self.params = training, eps, save_mean, save_var, res_space
            else:
                training, eps, batch_mean, batch_var, res_space = self.params
                out = torch.nn.functional.batch_norm(input_, batch_mean, 1/torch.square(batch_var), weight, bias, False, momentum, eps)
            return out

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            weight, bias, running_mean, running_var, output = stored
            training, eps, save_mean, save_var, res_space = self.params
            if training:
                di_app, dw_app, db_app = mybn_cpp.output_activated_bn_backward(
                    grad_output, output, weight, bias,
                    running_mean, running_var,
                    save_mean, save_var, training,
                    eps, [True, True,True])
            else:
                raise NotImplementedError

            if not nodel:
                del stored[4], stored[3], stored[2], stored[1], stored[0]

            return di_app, dw_app, db_app, None, None
