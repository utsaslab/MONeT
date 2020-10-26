import torch
from collections import OrderedDict
from monet.cvxpy_solver import Solution
from monet.schedule import Schedule, create_schedule


class MONeTWrapperF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, schedule, state_dict):
        input_detached = input_.detach()
        ctx.schedule = schedule
        ctx.save_for_backward(torch.ones(1))
        output = schedule.forward(input_detached, *state_dict)
        output.requires_grad_(True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        schedule = ctx.schedule
        schedule.backward(grad_output)
        return None, None, None


class MONeTWrapper(torch.nn.Module):
    def __init__(self, model, sol_file, input_shape):
        super(MONeTWrapper, self).__init__()
        if model._get_name() == 'MobileNetV2':
            model = torch.nn.Sequential(
                model.features,
                torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=1),
                model.classifier[0], model.classifier[1])
        self.model = model
        self.sol_file = sol_file
        self.input_shape = input_shape
        self.schedule = [create_schedule(model, sol_file, input_shape)]
        self.recreate_state_dict = False

    def train(self, mode=True):
        self.model.train(mode)

    def forward(self, x):
        x.requires_grad_(True)
        state_dict = self.state_dict(keep_vars=True)
        return MONeTWrapperF.apply(x, self.schedule[0], list(state_dict.values()))
