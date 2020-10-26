from .base import *
from .pack import *
import torch
import config
from time import time

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
class GreedyAlgoConvolution(OP):
    backward_storage = InputStorage(0, 1)
    params = None
    algorithm = -1
    fwd_algo = -1
    bwd_ip_algo = -1
    bwd_wt_algo = -1
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
                if self.fwd_algo == -1:
                    tfinal = -1
                    torch.cuda.empty_cache()
                    for alg in range(self.n_fwd_algos()):
                        try:
                            torch.cuda.reset_max_memory_allocated()
                            torch.cuda.synchronize()
                            t1 = time()
                            for it in range(10):
                                output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, alg)
                                del output
                            torch.cuda.synchronize()
                            t2 = time() - t1
                            if (t2 < tfinal or tfinal == -1) and torch.cuda.max_memory_allocated()/1024/1024/1024 < config.budget*1.01:
                                self.fwd_algo = alg
                                tfinal = t2
                            torch.cuda.empty_cache()
                            torch.cuda.reset_max_memory_allocated()
                        except Exception as e:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_max_memory_allocated()
                assert self.fwd_algo != -1
                try:
                    output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, self.fwd_algo)
                except Exception as e:
                    output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, -1)
                    self.fwd_algo = 4
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
                            if self.bwd_wt_algo == -1:
                                twtfinal = -1
                                torch.cuda.empty_cache()
                                for alg in range(self.n_bwd_wt_algos()):
                                    try:
                                        torch.cuda.reset_max_memory_allocated()
                                        torch.cuda.synchronize()
                                        t1 = time()
                                        for it in range(10):
                                            dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, alg)
                                        torch.cuda.synchronize()
                                        t2 = time() - t1
                                        if (t2 < twtfinal or twtfinal == -1) and torch.cuda.max_memory_allocated()/1024/1024/1024 < config.budget*1.01:
                                            self.bwd_wt_algo = alg
                                            twtfinal = t2
                                        del dw
                                        torch.cuda.empty_cache()
                                        torch.cuda.reset_max_memory_allocated()
                                    except Exception as e:
                                        torch.cuda.empty_cache()
                                        torch.cuda.reset_max_memory_allocated()
                            assert self.bwd_wt_algo != -1
                            try:
                                dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, self.bwd_wt_algo)
                            except Exception as e:
                                dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, -1)
                                self.bwd_wt_algo = 1

                        if do_grad[2]:
                            db = grad_output.sum([0, 2, 3])
                        del input_, input_detached

                if convtype == 1 or convtype == 2:
                    if weight is not None:
                        if do_grad[0]:
                            weight_detached = weight.detach()
                            if self.bwd_ip_algo == -1:
                                tipfinal = -1
                                torch.cuda.empty_cache()
                                for alg in range(self.n_bwd_ip_algos()):
                                    try:
                                        torch.cuda.reset_max_memory_allocated()
                                        torch.cuda.synchronize()
                                        t1 = time()
                                        for it in range(10):
                                            di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, alg)
                                            del di
                                        torch.cuda.synchronize()
                                        t2 = time() - t1
                                        if (t2 < tipfinal or tipfinal == -1) and torch.cuda.max_memory_allocated()/1024/1024/1024 < config.budget*1.01:
                                            self.bwd_ip_algo = alg
                                            tipfinal = t2
                                        torch.cuda.empty_cache()
                                        torch.cuda.reset_max_memory_allocated()
                                    except Exception as e:
                                        # print(e)
                                        torch.cuda.empty_cache()
                                        torch.cuda.reset_max_memory_allocated()
                            assert self.bwd_ip_algo != -1
                            weight_detached = weight.detach()
                            try:
                                di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, self.bwd_ip_algo)
                            except Exception as e:
                                di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, -1)
                                self.bwd_ip_algo = 1

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
                try:
                    output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, algorithm)
                except Exception as e:
                    output = conv_cpp.cudnn_convolution(input_, weight, padding, stride, dilation, groups, -1)
                    self.algorithm = 4
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
                            try:
                                dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, algo)
                            except Exception as e:
                                dw = conv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input_detached, padding, stride, dilation, groups, -1)
                                self.algorithm = convtype * 10 + 1
                        if do_grad[2]:
                            db = grad_output.sum([0, 2, 3])
                        del input_, input_detached

                if convtype == 1 or convtype == 2:
                    if weight is not None:
                        if do_grad[0]:
                            weight_detached = weight.detach()
                            try:
                                di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, algo)
                            except Exception as e:
                                di = conv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight_detached, padding, stride, dilation, groups, -1)
                                self.algorithm = convtype * 10 + 1

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
