import torch

from collections import namedtuple
import monet.lm_ops as lm_ops
import numpy as np

from gist.gist_graph import *
from monet.cvxpy_solver import *
from gist.gist_solver_info import *
from models.unet import UNet

ScheduleType = namedtuple('ScheduleType', 'recompute store_output store_intermediate')

class Schedule(Graph):
    def __init__(self, graph: Graph, info: SolverInfo):
        self.si = info
        self._nodes = graph.nodes
        self._outputs = graph._outputs
        self._op = []
        self._last_op_params = [None] * len(self.nodes)
        self._fwd_schedule = []
        self.real_mem = [[-1 for k in range(self.si.loss + 1)] for t in range(self.si.size - self.si.loss) ]

        # Stored tensors
        self._stored = [None] * len(self.nodes)
        self._stored_intermediate = [None] * len(self.nodes)
        self.deltensors = defaultdict(list)
        for j in self.si.nodes:
            if j<self.si.loss:
                next_node = j+1
                while isinstance(self.si.nodes[next_node], IntNode):
                    next_node = next_node + 1
                for i in self.si.nodes[j].local_tensors:
                    if j<self.si.size-1 and i not in self.si.nodes[next_node].local_tensors:
                        kk = self.si.solver_to_graph[i]
                        self.deltensors[j].append(kk)
        self.computeInstance = []
        self.transposeInstance = []
        for k, n in enumerate(self.nodes):
            if isinstance(n,ComputeNode):
                if n.op=="aten::t":
                    self.transposeInstance.append(k)
                else:
                    self.computeInstance.append(k)

        self._last_use_tensor = [0] * len(self.nodes)
        for k, n in enumerate(self.nodes):
            if isinstance(n, ComputeNode):
                for i, r in n.dependencies:
                    self._last_use_tensor[i] = max(k, self._last_use_tensor[i])
            else:
                self._last_use_tensor[k] = len(self.nodes)
        self._last_use_tensor[-1] = len(self.nodes)
        self._args = []
        self.lennodes = len(self.nodes)

    def init_schedule(self, mode):
        T = len(self.si.nodes) - self.si.loss
        # Create main structures
        self._op = [[] for i in range(T)]   # List of fwd operators
        self._bwd_op = []                   # List of bwd operators
        self._fwd_schedule = [[] for i in range(T)] # Forward schedule

        t = 0

        to_store_graph = [False for i in range(len(self._nodes))]
        for k, n in enumerate(self._nodes):
            if isinstance(n, ComputeNode) and n.op != "aten::t":
                p = 0
                op = lm_ops.list_ops(self.si.mode, n.op)[p]()
                storage_list = op.backward_storage
                if n.op == "aten::gist__convolution" or n.op == "aten::nosave_relu_":
                    continue    # do not store the input nosave_relu because already storing its compressed version
                if not isinstance(storage_list, list):
                    storage_list = [storage_list]
                for storage in storage_list:
                    if isinstance(storage, lm_ops.InputStorage):
                        for i in storage.ids:
                            to_store_graph[n.args[i].index] = True
                    elif isinstance(storage, lm_ops.OutputStorage):
                        to_store_graph[k] = True

        for i, n in enumerate(self._nodes):
            if isinstance(n, ComputeNode) and n.op != "aten::t":
                j = self.si.graph_to_solver[i]
                p = 0
                op = lm_ops.list_ops(self.si.mode, n.op)[p]()
                if "relu_" in n.op or "hardtanh_" in n.op or "add_" in n.op:
                    op.inplace = True
                if n.is_depthwise:
                    op.is_depthwise = True
                self._op[t].append(op)
                schedule_intermediate = False
                storage = op.backward_storage
                if not isinstance(storage, list):
                    storage = [storage]
                for store in storage:
                    if isinstance(store, lm_ops.IntermediateStorage):
                        schedule_intermediate = True
                self._fwd_schedule[0].append(ScheduleType(True, to_store_graph[i], schedule_intermediate))
            else:
                # Node represents a parameter
                self._op[t].append(None)
                self._fwd_schedule[t].append(None)

        # Initialize backward pass structures
        for t in range(T):
            bwd_t = self.si.loss + t
            if t!=0 and isinstance(self.si.nodes[bwd_t], BwdNode):
                n = self.nodes[self.si.solver_to_graph[self.si.bwd_to_fwd[bwd_t]]] # Backportability to when si didn't support depthwise
                p = 0
                op = lm_ops.list_ops(self.si.mode, n.op)[p]()
                if n.is_depthwise:
                    op.is_depthwise = True
                self._bwd_op.append(op)
            else:
                self._bwd_op.append(None)


    def _forward(self, schedule, keep_tensors=[], t=0):
        tensors = self._stored
        # Only store tensors we need
        self._stored = [None] * self.lennodes
        self.compressed_data = [None] * self.lennodes
        for k, n in enumerate(self._nodes):
            if k in self.computeInstance:
                assert self._op[t][k] is not None
                recompute, store_output, store_intermediate = False, False, False
                if k < len(schedule):
                    recompute, store_output, store_intermediate = schedule[k]
                assert k == n.id
                assert(tensors[k] is None)
                args = [a.value if isinstance(a, ComputeNode.V) else tensors[a.index].requires_grad_(a.requires_grad) for a in n.args]

                # Run forward
                if store_intermediate:
                    tensors[k], self._stored_intermediate[k] = self._op[t][k].forward(*args)
                    assert self._stored_intermediate[k] is not None
                elif n.op == "aten::nosave_relu_":
                    tensors[k], self.compressed_data[k] = self._op[t][k].forward(*args)
                    assert self.compressed_data[k] is not None
                else:
                    tensors[k] = self._op[t][k].forward(*args)

                if hasattr(self._op[t][k], 'params'):
                    self._last_op_params[k] = self._op[t][k].params
                if isinstance(self._op[t][k], lm_ops.NativeOP):
                    self._last_op_params[k] = self._op[t][k]
                del args
                if hasattr(self._op[t][k],'inplace'):
                    if self._op[t][k].inplace:
                        j = self.si.graph_to_solver[n.id]
                        j_old = self.si.nodes[j].args[0][0]
                        tensors[self.si.solver_to_graph[j_old]] = None
                        self._stored[self.si.solver_to_graph[j_old]] = None
                # delete relu output in gist conv or gist mp
                if (n.op == "aten::gist__convolution" or n.op == "aten::gist_max_pool2d"):
                    relu_tensor_idx = n.args[0].index
                    j = self.si.graph_to_solver[n.id]
                    if relu_tensor_idx in self.deltensors[j]:
                        tensors[relu_tensor_idx] = None
                        self._stored[relu_tensor_idx] = None

                j = self.si.graph_to_solver[n.id]
                for kk in self.deltensors[j]:
                    if kk not in keep_tensors:
                        tensors[kk] = None

                if store_output:
                    assert tensors[k] is not None, 'Storing a None tensor for node graph node %d, solver node %d, op %s ' % (k, self.si.graph_to_solver[k], n.op)
                    self._stored[k] = tensors[k].half()

                if store_intermediate:
                    assert recompute

            elif k in self.transposeInstance:
                pass
            else:
                # Use input and check tensor shape
                tensors[k] = self._args[k]
                self._stored[k] = tensors[k]
        return tensors

    def forward(self, *args):
        self._args = args
        tensors = self._forward(self._fwd_schedule[0], t=0)
        r = tensors[-1]
        del tensors
        return r

    def backward(self, delta_x):
        bw_tensors = [None] * self.lennodes
        for opk in self._outputs:
            bw_tensors[opk] = delta_x
        T = len(self.si.nodes) - self.si.loss
        tensors = self._stored
        self._stored = [None] * self.lennodes
        solver_min = self.si.solver_to_graph[0]

        for k, n in reversed(list(enumerate(self._nodes))):
            if isinstance(n, ComputeNode):
                if n.op == "aten::t":
                    continue
                bwd_t = self.si.fwd_to_bwd[self.si.graph_to_solver[k]]
                t_start = bwd_t - self.si.loss
                t = t_start
                # Get the grad_fn
                grad_nd = self._bwd_op[t]
                assert grad_nd is not None

                # Find all input/output dependencies
                required_storage = []

                storage_list = grad_nd.backward_storage
                if not isinstance(storage_list, list):
                    storage_list = [storage_list]

                for storage in storage_list:
                    if isinstance(storage, lm_ops.InputStorage):
                        for i in storage.ids:
                            required_storage.append(n.args[i].index)
                    elif isinstance(storage, lm_ops.OutputStorage):
                        required_storage.append(k)

                if hasattr(grad_nd, 'params'):
                    grad_nd.params = self._last_op_params[k]
                if isinstance(grad_nd, lm_ops.NativeOP):
                    assert self._last_op_params[k] is not None
                    grad_nd = self._last_op_params[k]
                assert grad_nd is not None

                # Assemble the forward inputs
                stored = []
                idx = 0
                for storage in storage_list:
                    if isinstance(storage, lm_ops.InputStorage):
                        for sidx in range(idx, idx+len(storage.ids)):
                            i = required_storage[sidx]
                            if n.compress_conv == i and tensors[i] is None:
                                torch.cuda.synchronize()
                                tensors[i] = grad_nd.uncompress(*self.compressed_data[i])
                                self.compressed_data[i] = None
                            assert tensors[i] is not None, "Tensor dependency %d not computed or stored for node %d %s t=%d" % (i, k, n.op, t)
                            stored.append(tensors[i].float())
                            if i >= solver_min:
                                if self.si.last_use_bwd[i]["ip"] == k:
                                    tensors[i] = None
                        idx = sidx + 1
                    elif isinstance(storage, lm_ops.OutputStorage):
                        i = required_storage[idx]
                        stored.append(tensors[i].float())
                        if self.si.last_use_bwd[i]["ip"] == k:
                            tensors[i] = None
                        idx = idx + 1
                    elif isinstance(storage, lm_ops.IntermediateStorage):
                        assert self._stored_intermediate[k] is not None, "Intermediate output not computed for node %d" % k
                        stored.append(self._stored_intermediate[k])
                        self._stored_intermediate[k] = None

                # Call backward
                assert bw_tensors[k] is not None, "Backward input not computed for node %d (graph), %d (solver), %s (node)" % (k, self.si.graph_to_solver[k], self.si.nodes[self.si.graph_to_solver[k]])
                bw_outs = grad_nd.backward(bw_tensors[k], stored)

                if not isinstance(bw_outs, (list, tuple)):
                    bw_outs = (bw_outs,)

                assert len(bw_outs) == len(n.dependencies), \
                    "Require the same number of grad outputs as forward inputs" \
                    " %s (%d) , %s (%d) %s" % (
                        repr(bw_outs), len(bw_outs),
                        repr(n.dependencies), len(n.dependencies), n)

                # Free the backward tensor
                bw_tensors[k] = None

                # Accumulate the backward gradient
                for (i, r), o in zip(n.dependencies, bw_outs):
                    if r:
                        if o is not None:
                            if bw_tensors[i] is None:
                                bw_tensors[i] = o
                            else:
                                bw_tensors[i] += o
                del grad_nd

            elif self._args[k].requires_grad:
                self._args[k].backward(bw_tensors[k])
                bw_tensors[k] = None

        # Clear params for BN
        for k, n in enumerate(self._nodes):
            if k in self.computeInstance and n.op == "aten::batch_norm":
                self._op[0][k].params = None

if __name__ == '__main__':
    import argparse
    import torchvision
    from time import time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('bs')
    parser.add_argument('budget')
    parser.add_argument(
    "--check_runtime", action="store_true",
    help="Compute the runtime difference between gist and normal model.")
    args = parser.parse_args()

    budget = float(args.budget)
    bs = int(args.bs)
    model_name = args.model.split(".")[-1][:-2]
    mode = "gist"
    print("Batch size ", bs)
    print("Model", model_name)
    print("Mode", mode)

    # Initialize pool of budget
    pool_shape = ( int(budget * 256 * 1024 * 1024) >> 3 ) << 3
    t = torch.zeros(pool_shape).cuda()
    del t
    torch.cuda.reset_max_memory_allocated()

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

    graph = Graph.create(model, input_shape=(3, height, width))
    model.cuda()

    input_ = torch.randn((bs, 3, height, width)).cuda()
    solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
    solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
    schedule = Schedule(graph, solver_info)
    schedule.init_schedule(mode)
    torch.cuda.synchronize()
    torch.cuda.reset_max_memory_allocated()

    if args.check_runtime:
        start_event_monet = torch.cuda.Event(enable_timing=True)
        end_event_monet = torch.cuda.Event(enable_timing=True)
        for iterid in range(120):
            if iterid == 100:
                torch.cuda.reset_max_memory_allocated()
                start_event_monet.record()
            x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))
            schedule.backward(-torch.ones_like(x1))
            for v in model.parameters():
                v.grad = None
        end_event_monet.record()
        torch.cuda.synchronize()
        del x1
        monet_maxmem = torch.cuda.max_memory_allocated() / 2**20

        print("monet: %f ms avg, %8.2f MB" % (start_event_monet.elapsed_time(end_event_monet)/20, monet_maxmem))
    else:
        print("simple fwd bwd")
        x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))
        schedule.backward(-torch.ones_like(x1))
        torch.cuda.synchronize()
        print("Max mem: %.3f MB" % (torch.cuda.max_memory_allocated()/1024/1024))
    print("Done")
