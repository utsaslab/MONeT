from .base import *
from .pack import *
import torch

conv_cpp = load(name="conv_cpp", sources=[this_dir/"conv.cpp"], extra_cflags=['-std=c++17'], extra_include_paths=[str(this_dir)], with_cuda=True)


@implements(['aten::_convolution'], ['none'])
class PytorchConvolution(OP):
    backward_storage = InputStorage(0, 1)

    def forward(self, input_, weight, bias, stride, padding, dilation, transposed, output_padding, groups, *args, **kwargs):
        assert not transposed
        self.params = stride, padding, dilation, groups, input_.shape, weight.shape
        self.do_grad = [input_.requires_grad, weight.requires_grad, bias is not None and bias.requires_grad]
        with torch.no_grad():
            output = conv_cpp.forward_normal(input_, weight, stride, padding, dilation, groups)
            return output + bias.view((1, -1, 1, 1))

    def backward(self, grad_output, stored):
        input_, weight = stored
        stride, padding, dilation, groups, input_shape, weight_shape = self.params
        di = dw = db = None
        if input_ is not None:
            if self.do_grad[1]:
                dw = conv_cpp.backward_weight_normal(weight_shape, grad_output, input_, stride, padding, dilation, groups)

        if weight is not None:
            if self.do_grad[0]:
                di = conv_cpp.backward_input_normal(input_shape, grad_output, weight, stride, padding, dilation, groups)

        if self.do_grad[2]:
            db = grad_output.sum([0, 2, 3])

        return di, dw, db


@implements(['aten::_convolution'], ['normal', 'multiway_newnode', 'multiway', 'newnode'])
class DefaultAlgoConvolution(OP):
    backward_storage = InputStorage(0, 1)
    params = None
    algorithm = -1
    is_depthwise = False

    def forward(self, input_, weight, bias, stride, padding, dilation, transposed, output_padding, groups, *args, **kwargs):
        with torch.no_grad():
            assert not transposed
            self.params = stride, padding, dilation, groups, input_.shape, weight.shape, [input_.requires_grad, weight.requires_grad, bias is not None and bias.requires_grad]
            algorithm = -1
            if self.is_depthwise:
                if bias == None:
                    return conv_cpp.convolution_main(input_, weight, torch.tensor(1), stride, padding, dilation, transposed, output_padding, groups)
                return conv_cpp.convolution_main(input_, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            else:
                input_ = input_.detach()
                weight = weight.detach()
                output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, algorithm)
                if bias is not None:
                    output[:] += bias.view((1, -1, 1, 1))
            return output

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            input_, weight = stored
            stride, padding, dilation, groups, input_shape, weight_shape, do_grad = self.params
            algorithm = self.algorithm
            algo = -1
            convtype = algorithm // 10
            # Delete the stored inputs
            if not nodel:
                del stored[1]
                del stored[0]
            di = dw = db = None

            if self.is_depthwise:
                di, dw = conv_cpp.backward_depthwise(grad_output, input_, weight, (weight.shape[2], weight.shape[3]), stride, padding, dilation, do_grad[:2])
            else:
                if convtype == 0 or convtype == 2:
                    if input_ is not None:
                        if do_grad[1]:
                            input_detached = input_.detach()
                            dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, algo)
                        if do_grad[2]:
                            db = grad_output.sum([0, 2, 3])
                        del input_, input_detached

                if convtype == 1 or convtype == 2:
                    if weight is not None:
                        if do_grad[0]:
                            weight_detached = weight.detach()
                            di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, algo)

            if do_grad[2]:
                return di, dw, db
            return di, dw

@implements(['aten::_convolution'], ['conv_multiway_newnode', 'conv_normal'])
class SpecificAlgoConvolution(OP):
    backward_storage = InputStorage(0, 1)
    params = None
    algorithm = -1
    is_depthwise = False

    def forward(self, input_, weight, bias, stride, padding, dilation, transposed, output_padding, groups, *args, **kwargs):
        with torch.no_grad():
            assert not transposed
            self.params = stride, padding, dilation, groups, input_.shape, weight.shape, [input_.requires_grad, weight.requires_grad, bias is not None and bias.requires_grad]
            if self.is_depthwise:
                if bias == None:
                    return conv_cpp.convolution_main(input_, weight, torch.tensor(1), stride, padding, dilation, transposed, output_padding, groups)
                return conv_cpp.convolution_main(input_, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            else:
                algorithm = self.algorithm
                input_ = input_.detach()
                weight = weight.detach()
                output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, algorithm)
                if bias is not None:
                    output[:] += bias.view((1, -1, 1, 1))
            return output

    def backward(self, grad_output, stored, nodel=False):
        with torch.no_grad():
            input_, weight = stored
            stride, padding, dilation, groups, input_shape, weight_shape, do_grad = self.params
            algorithm = self.algorithm
            algo = algorithm % 10
            convtype = algorithm // 10
            # Delete the stored inputs
            if not nodel:
                del stored[1]
                del stored[0]
            di = dw = db = None

            if self.is_depthwise:
                di, dw = conv_cpp.backward_depthwise(grad_output, input_, weight, (weight.shape[2], weight.shape[3]), stride, padding, dilation, do_grad[:2])
            else:
                if convtype == 0 or convtype == 2:
                    if input_ is not None:
                        if do_grad[1]:
                            input_detached = input_.detach()
                            dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, algo)
                        if do_grad[2]:
                            db = grad_output.sum([0, 2, 3])
                        del input_, input_detached

                if convtype == 1 or convtype == 2:
                    if weight is not None:
                        if do_grad[0]:
                            weight_detached = weight.detach()
                            di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, algo)

            if do_grad[2]:
                return di, dw, db
            return di, dw

    @staticmethod
    def n_fwd_algos():
        return conv_cpp.n_fwd_algos()

    @staticmethod
    def n_bwd_ip_algos():
        return conv_cpp.n_bwd_ip_algos()

    @staticmethod
    def n_bwd_wt_algos():
        return conv_cpp.n_bwd_wt_algos()
