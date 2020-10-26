# Get runtime of original PyTorch model
import argparse
import torch, torchvision
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('bs')
args = parser.parse_args()

bs = int(args.bs)
model_name = args.model.split(".")[-1][:-2]
print("Batch size ", bs)
print("Model", model_name)

if args.model == 'unet':
    height, width = 416, 608
    model = UNet(n_channels=3, n_classes=1, height=height, width=width)
else:
    height, width = 224, 224
    model = eval(args.model, {'torch': torch, 'torchvision': torchvision})

if 'mobilenet_v2' in args.model:
    model = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=1),
        model.classifier[0], model.classifier[1])
model.cuda()
input_ = torch.randn((bs, 3, height, width)).cuda()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()

torch.backends.cudnn.benchmark = True

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
if 'googlenet' in args.model:
    for i in range(120):
        if i==100:
            start_event.record()
        x0 = model(input_)
        (x0[0] + x0[1] + x0[2]).sum().backward()
        del x0
    end_event.record()
    torch.cuda.synchronize()
else:
    for i in range(120):
        if i==100:
            start_event.record()
        x0 = model(input_)
        x0.sum().backward()
        del x0
    end_event.record()
    torch.cuda.synchronize()
orig_maxmem = torch.cuda.max_memory_allocated() / 2**20
print("original: %fms avg, %8.2f MB" % (start_event.elapsed_time(end_event)/20, orig_maxmem))
del model