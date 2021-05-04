This folder contains the code for our implementation of [Gist](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf) in PyTorch, used for comparison with MONeT.

`gist_graph.py` creates the modified forward pass graph for Gist by adding the intermediate encodings for ReLU->MaxPool layers and annotating ReLU->Conv layers for the SSDC technique in Gist. 

`gist_solver_info.py` adds the backward pass information to the graph.

`gist_schedule.py` runs a Gist schedule.


It can be run using:

```
python gist_schedule.py [MODEL] [BATCH_SIZE] [BUDGET in GB] --check_runtime
```

Example,
```
python gist_schedule.py "torchvision.models.googlenet()" 320 14 --check_runtime
```
