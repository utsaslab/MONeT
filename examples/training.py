import torch, torchvision
from monet.cvxpy_solver import Solution
from monet.monet_wrapper import MONeTWrapper
import time

input = torch.randn(184,3,224,224).cuda()
model = torchvision.models.resnet50()
input_shape = (3,224,224)

# Can change to use absolute path instead of relative
sol_file = "../data/monet_r50_184_24hr/solution_resnet50_184_inplace_conv_multiway_newnode_10.00.pkl"
train_model = MONeTWrapper(model, sol_file, (3,224,224)).cuda()
output = train_model(input)
output.sum().backward()
print("Memory used: %6.2f MB" % (torch.cuda.max_memory_allocated()/1024/1024))

