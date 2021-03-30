# MONeT: Memory Optimization for Deep Networks

Implemented over PyTorch, MONeT schedules allow training deep networks on a constrained memory budget with minimal computational overhead. MONeT jointly determines checkpointing as well as operator implementations, reducing GPU memory by as much as 3x with a compute overhead of 9-16%.

<div align="center">
<img src="figs/monet_concept_fig.jpeg" width="800">
</img></div>

> **Memory Optimization for Deep Networks** <br/>
> Aashaka Shah, Chao-Yuan Wu, Jayashree Mohan, Vijay Chidambaram, Philipp Kr&auml;henb&uuml;hl <br/>
> In ICLR 2021 [[paper]](https://openreview.net/pdf?id=bnY0jm4l59)


## Installation
MONeT has been tested with PyTorch 1.5.1, torchvision 0.6.1, and cudatoolkit 10.1. Create a conda environment with python 3.7 or greater. Inside the environment, install the following packages: `cvxpy`, `gurobi`, `pandas`, `ninja-build`, `coinor-cbc`, `coinor-libcbc-dev`, `cylp`.

[install.sh](install.sh) provides the installation script.

Clone this repo and install the package. Ensure that the conda environment is activated.
```
git clone --recursive https://github.com/utsaslab/MONeT
cd MONeT
pip install -e .
```

## Getting Started

### MONeT usage
MONeT has been tested for single-GPU training and single-machine multi-GPU Distributed Data Parallel training. To get started with MONeT using solutions in the schedule zoo, add the following imports to your code:

```
from monet.cvxpy_solver import Solution
from monet.monet_wrapper import MONeTWrapper
```

Wrap your model using a MONeTWrapper
```
monet_model = MONeTWrapper(model, solution_file, input_shape)
```

Use the model like you normally would
```
output = monet_model(input) # Forward pass
output.sum().backward() # Backward pass
```

A working version of this code can be found at [examples/training.py](examples/training.py).

For Distributed Data Parallel training, `monet_model` can be wrapped by `torch.nn.parallel.DistributedDataParallel` like any other model.
A working distributed training code can be found at [examples/dist_training.py](examples/dist_training.py).

The [examples/imagenet.py](examples/imagenet.py) has been modified to use MONeT schedules for ImageNet training.
``` 
python imagenet.py DATA_DIR -a [arch] --gpu 0 \
        --epochs [num_epochs] \
        --batch-size [batch_size] \
        --solution_file [path to solution file]
```

At higher batch sizes, it is possible that the PyTorch memory allocator outputs an Out-Of-Memory error even if the schedule executed should run without any issues. This is because of the caching-nature of the memory allocator. Please create a pool of expected memory usage before allocating any tensors for the training using the following code snippet:

```
pool = torch.zeros(expected_memory/4).cuda()
del pool
```

## Schedule zoo
We have already created some schedules which can be used right off the bat.
Simply install MONeT, modify your training similar to [examples/imagenet.py](examples/imagenet.py), and use the memory efficient schedules for training!
The schedule zoo is hosted in the `data` directory.
You can use the results below to pick the right schedule according to your requirements.

A solution [solution_resnet50_184_inplace_conv_multiway_newnode_10.00.pkl](https://github.com/aashaka/monet-schedules/blob/master/monet_r50_184_24hr/solution_resnet50_184_inplace_conv_multiway_newnode_10.00.pkl) uses 10 GB memory for training ResNet-50 with a batch size of 184, and according to the results, has a 3.22% overhead over the original PyTorch implementation which uses 15.06 GB memory.

## Results

| ResNet-50 (184) | Memory (GB) | Compute Overhead (%) |
|-----------|-------------|----------------------|
| PyTorch   | 15.06       | 0                    |
| MONeT     | 10.01       | 3.22%                |
| MONeT     | 9.01        | 4.68%                |
| MONeT     | 8.01        | 5.56%                |
| MONeT     | 6.99        | 7.28%                |
| MONeT     | 6.00        | 9.31%                |
| MONeT     | 4.99        | 11.95%               |

| GoogleNet (320) | Memory (GB) | Compute Overhead (%) |
|-----------|-------------|----------------------|
| PyTorch   | 14.93       | 0                    |
| MONeT     | 9.98        | 7.13%                |
| MONeT     | 8.99        | 7.87%                |
| MONeT     | 8.01        | 8.44%                |
| MONeT     | 7.02        | 9.71%                |
| MONeT     | 6.01        | 12.14%               |
| MONeT     | 4.99        | 15.77%               |

| UNet (11)   | Memory (GB) | Compute Overhead (%) |
|---------|-------------|----------------------|
| PyTorch | 14.32       | 0                    |
| MONeT   | 10.01       | -4.10%               |
| MONeT   | 9.01        | -2.07%               |
| MONeT   | 8.02        | -0.09%               |
| MONeT   | 7.00        | 1.39%                |
| MONeT   | 6.01        | 4.95%                |
| MONeT   | 5.01        | 11.51%               |

| Mobilenet (272) | Memory (GB) | Compute Overhead (%) |
|-----------|-------------|----------------------|
| PyTorch   | 14.46       | 0                    |
| MONeT     | 10.02       | 2.40%                |
| MONeT     | 9.01        | 3.10%                |
| MONeT     | 8.02        | 4.77%                |
| MONeT     | 7.01        | 5.53%                |
| MONeT     | 6.01        | 7.55%                |
| MONeT     | 5.01        | 8.72%                |

| VGG-16 (176) | Memory (GB) | Compute Overhead (%) |
|---------|-------------|----------------------|
| PyTorch | 14.12       | 0                    |
| MONeT   | 9.71        | -5.30%               |
| MONeT   | 8.66        | -4.64%               |
| MONeT   | 7.88        | -2.18%               |
| MONeT   | 6.82        | 1.99%                |
| MONeT   | 5.90        | 5.44%                |
| MONeT   | 5.51        | 9.11%                |


## Advanced MONeT usage
Obtain the Gurobi academic license from the Gurobi [website](https://www.gurobi.com/downloads/end-user-license-agreement-academic/). Login with a .edu email to get the free license.

1. To create a MONeT solution:
```
python cvxpy_solver.py MODEL BATCH_SIZE BUDGET MODE "GUROBI" --time_limit TIME_LIMIT
```

MODEL format: `"torchvision.models.<model>()"`. For UNeT, the format is `"unet"`. <br/>
BUDGET is the memory budget in GB <br/>
MODE is "inplace_conv_multiway_newnode" for complete MONeT <br/>
TIME_LIMIT is the solver time limit in seconds <br/>
The flag `--ablation` can be added to disable checkpointing when creating a solution.

2. To profile a MONeT schedule given a solution:
```
python schedule.py MODEL BATCH_SIZE BUDGET MODE "GUROBI" \
        --run_bs --solution_file SOLUTION_FILE
```
The flag `--run_bs` can be replaced by `--check_runtime` to check the runtime of the schedule or `--check_diff` to check the gradients of MONeT against original PyTorch.


Other modes may be used for experimenting with MONeT:
- `inplace_` prefix enables operator optimization
- `conv_normal` selects conv-optimization
- `multiway` selects output-activated optimization
- `newnode` selects intermediate-activate optimization

Refer the paper for details about the optimizations.

## Citation
If you use MONeT in your work, please consider citing us as

```
@misc{shah2020memory,
      title={Memory Optimization for Deep Networks},
      author={Aashaka Shah and Chao-Yuan Wu and Jayashree Mohan and Vijay Chidambaram and Philipp Krähenbühl},
      year={2020},
      eprint={2010.14501},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
The code for UNeT is taken from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) by [milesial](https://github.com/milesial). Distributed Data Parallel training example code is borrowed from the [distributed tutorial](https://github.com/yangkky/distributed_tutorial) by [yangkky](https://github.com/yangkky).
