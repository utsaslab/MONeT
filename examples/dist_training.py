import os
import time
import argparse
import torch.multiprocessing as mp
import torchvision
import torch
import torch.distributed as dist
from monet.cvxpy_solver import Solution
from monet.monet_wrapper import MONeTWrapper

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    torch.manual_seed(0)
    sol_file = "../data/monet_r50_184_24hr/solution_resnet50_184_inplace_conv_multiway_newnode_10.00.pkl"
    model = MONeTWrapper(torchvision.models.resnet50(), sol_file, (3,224,224))
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 184
    input_ = torch.randn((batch_size,3,224,224)).cuda(gpu)

    # Wrap the model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    for i in range(100):
        if i == 80:
            torch.cuda.synchronize()
            t0 = time.time()
        output = model(input_)
        output.sum().backward()
    torch.cuda.synchronize()
    print(gpu, "time:", time.time()-t0, "memory:", torch.cuda.max_memory_allocated(gpu)/1024/1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    main()