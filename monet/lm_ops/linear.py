from .base import *
import torch


@implements(['aten::addmm'], ['none', 'gist'])
class AddMM(OP):
    backward_storage = InputStorage(1, 2)
    params = None

    def forward(self, bias, input_, weight, beta, alpha):
        # Just don't want to deal with this right now
        assert alpha == 1
        assert beta == 1
        with torch.no_grad():
            return torch.addmm(bias, input_, weight.t(), beta=beta, alpha=alpha)

    def backward(self, output_grad, stores):
        with torch.no_grad():
            input_ = stores[0]
            weight = stores[1]
            del stores[1]
            del stores[0]
            return output_grad.sum(0), output_grad.mm(weight), output_grad.t().mm(input_)

@implements(['aten::addmm'], ['normal', 'multiway', 'newnode', 'multiway_newnode', 'conv_multiway_newnode', 'conv_normal'])
class LowMemAddMM(OP):
    backward_storage = InputStorage(1, 2)
    params = None
    algorithm = -1

    def forward(self, bias, input_, weight, beta, alpha):
        # NOTE: bias will never be None because otherwise the trace would have generated matmul instead of addmm
        self.params = [input_.requires_grad, weight.requires_grad, bias is not None and bias.requires_grad]
        # Just don't want to deal with this right now
        assert alpha == 1
        assert beta == 1
        with torch.no_grad():
            return torch.addmm(bias, input_, weight.t(), beta=beta, alpha=alpha)

    def backward(self, output_grad, stores, nodel=False):
        with torch.no_grad():
            addmmtype = self.algorithm // 10
            input_ = stores[0]
            weight = stores[1]
            if not nodel:
                del stores[1]
                del stores[0]
            di = dw = db = None
            if addmmtype == 0 or addmmtype == 2:
                if input_ is not None:
                    if self.params[1]:
                        dw = output_grad.t().mm(input_)
                    if self.params[2]:
                        db = output_grad.sum(0)
                    del input_

            if addmmtype == 1 or addmmtype == 2:
                if weight is not None:
                    if self.params[0]:
                        di = output_grad.mm(weight)

            return db, di, dw