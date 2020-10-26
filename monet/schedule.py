import torch

from collections import namedtuple
import monet.lm_ops as lm_ops
import numpy as np

from monet.graph import *
from monet.cvxpy_solver import *
from monet.solver_info import *
from models.unet import UNet

ScheduleType = namedtuple('ScheduleType', 'recompute store_output store_intermediate')

class Schedule(Graph):
    def __init__(self, graph: Graph, info: SolverInfo):
        self.si = info
        self._nodes = graph.nodes
        self._outputs = graph._outputs
        self.solution = None
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

    def init_schedule(self, solution: Solution, mode):
        self.solution = solution
        T = len(self.si.nodes) - self.si.loss
        # Create main structures
        self._op = [[] for i in range(T)]   # List of fwd operators
        self._bwd_op = []                   # List of bwd operators
        self._fwd_schedule = [[] for i in range(T)] # Forward schedule

        # Initialize forward pass structures
        for t in range(T):
            for i, n in enumerate(self._nodes):
                if isinstance(n, ComputeNode) and n.op != "aten::t":
                    j = self.si.graph_to_solver[i]
                    p = 0
                    if self.si.compute_newnode and (self.si.nodes[j].has_intermediates):
                        if self.si.nodes[j].has_intermediates:
                            for (p_option, intid) in self.si.nodes[j].intermediates:
                                if solution.r[t][intid]:
                                    assert solution.r[t][j]
                                    p = p_option
                    op = lm_ops.list_ops(self.si.mode, n.op)[p]()
                    if self.si.select_conv_algo and j in self.si.conv_list:
                        num_fwd_algos = op.n_fwd_algos()
                        for c in range(num_fwd_algos):
                            if solution.rf[t,self.si.conv_list[j], c]:
                                op.algorithm = c
                                break
                    if n.is_depthwise:
                        op.is_depthwise = True
                    if self.si.do_inplace and j in self.si.inplace_list:
                        if solution.ip[t][j]:
                            op.inplace = True
                    self._op[t].append(op)
                    schedule_intermediate = False
                    storage = op.backward_storage
                    if not isinstance(storage, list):
                        storage = [storage]
                    for store in storage:
                        if isinstance(store, lm_ops.IntermediateStorage):
                            schedule_intermediate = True
                    if t < T - 1:
                        s = solution.s[t+1][j]
                        r = solution.r[t][j]
                        self._fwd_schedule[t].append(ScheduleType(r, s, schedule_intermediate))
                    else:
                        r = solution.r[t][j]
                        self._fwd_schedule[t].append(ScheduleType(r, False, schedule_intermediate))
                else:
                    # Node represents a parameter
                    self._op[t].append(None)
                    self._fwd_schedule[t].append(None)

        # Initialize backward pass structures
        for t in range(T):
            bwd_t = self.si.loss + t
            if t!=0 and isinstance(self.si.nodes[bwd_t], BwdNode):
                n = self.nodes[self.si.solver_to_graph[self.si.bwd_to_fwd[bwd_t]]] # Backportability to when si didn't support depthwise
                options = len(solution.m[t])
                p = 0
                for o in range(options):
                    if solution.m[t][o] == 1:
                        p = o
                op = lm_ops.list_ops(self.si.mode, n.op)[p]()
                if n.is_depthwise:
                    op.is_depthwise = True
                if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op=="aten::addmm":
                    algo = 0
                    if self.si.nodes[bwd_t].bwd_op =="param_grad":
                        algo_type = 0
                    else:
                        algo_type = 1
                    if self.si.select_conv_algo and bwd_t in self.si.conv_list:
                        algo = -1
                        num_algos = solution.rf.shape[2]
                        for c in range(num_algos):
                            if solution.rf[t,self.si.conv_list[bwd_t], c]:
                                algo = c
                                break
                        if algo == -1:
                            raise RuntiimeError("Algorithm not decided", t, bwd_t)
                    algo = algo_type*10 + algo
                    op.algorithm = algo
                self._bwd_op.append(op)
            else:
                self._bwd_op.append(None)


    def _forward(self, schedule, keep_tensors=[], t=0):
        tensors = self._stored
        # Only store tensors we need
        self._stored = [None] * self.lennodes
        for k, n in enumerate(self._nodes):
            if k in self.computeInstance:
                assert self._op[t][k] is not None
                recompute, store_output, store_intermediate = False, False, False
                if k < len(schedule):
                    recompute, store_output, store_intermediate = schedule[k]
                assert k == n.id
                if recompute:
                    # Recompute only if the output does not exist or if it exists but we have to compute intermediate node
                    if tensors[k] is None or store_intermediate:
                        if t>0 and n.op == "aten::batch_norm":
                            self._op[t][k].params = self._last_op_params[k]
                        args = [a.value if isinstance(a, ComputeNode.V) else tensors[a.index].requires_grad_(a.requires_grad) for a in n.args]

                        # Run forward
                        if store_intermediate:
                            tensors[k], self._stored_intermediate[k] = self._op[t][k].forward(*args)
                            assert self._stored_intermediate[k] is not None
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

                if store_output:
                    assert tensors[k] is not None, 'Storing a None tensor for node graph node %d, solver node %d, op %s ' % (k, self.si.graph_to_solver[k], n.op)
                    self._stored[k] = tensors[k]

                if store_intermediate:
                    assert recompute

                j = self.si.graph_to_solver[n.id]
                for kk in self.deltensors[j]:
                    if kk not in keep_tensors:
                        tensors[kk] = None

            elif k in self.transposeInstance:
                pass
            else:
                # Use input and check tensor shape
                tensors[k] = self._args[k]
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

        for k, n in reversed(list(enumerate(self._nodes))):
            if isinstance(n, ComputeNode):
                if n.op == "aten::t":
                    continue
                bwd_t = self.si.fwd_to_bwd[self.si.graph_to_solver[k]]
                t_start = bwd_t - self.si.loss
                t_end = t_start
                if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op=="aten::addmm":
                    t_start = t_start-1

                for t in range(t_start,t_end+1):
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

                    # Recompute the inputs
                    if (t_start < t_end and t == t_end):
                        tensors = self._forward(self._fwd_schedule[t], [required_storage[1]], t=t)
                    else:
                        tensors = self._forward(self._fwd_schedule[t], required_storage, t)
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
                                if not(t_start < t_end and t == t_end) or sidx==idx+len(storage.ids)-1:
                                    assert tensors[i] is not None, "Tensor dependency %d not computed or stored for node %d %s t=%d" % (i, k, n.op, t)
                                stored.append(tensors[i])
                            idx = sidx + 1
                        elif isinstance(storage, lm_ops.OutputStorage):
                            i = required_storage[idx]
                            stored.append(tensors[i])
                            idx = idx + 1
                        elif isinstance(storage, lm_ops.IntermediateStorage):
                            assert self._stored_intermediate[k] is not None, "Intermediate output not computed for node %d" % k
                            stored.append(self._stored_intermediate[k])
                            self._stored_intermediate[k] = None

                    # Free the forward tensors
                    del tensors

                    # Call backward
                    assert bw_tensors[k] is not None, "Backward input not computed for node %d (graph), %d (solver), %s (node)" % (k, self.si.graph_to_solver[k], self.si.nodes[self.si.graph_to_solver[k]])
                    bw_outs = grad_nd.backward(bw_tensors[k], stored)

                    if not isinstance(bw_outs, (list, tuple)):
                        bw_outs = (bw_outs,)

                    assert len(bw_outs) == len(n.dependencies), \
                        "Require the same number of grad outputs as forward inputs" \
                        " %s (%d) , %s (%d)" % (
                            repr(bw_outs), len(bw_outs),
                            repr(n.dependencies), len(n.dependencies))

                    # Free the backward tensor
                    if t == t_end:
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

def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

def load_solution(filename):
    from monet.cvxpy_solver import Solution
    import pickle
    print(f'Loading solver_info, solution from {filename}')
    with open(filename, 'rb') as f:
        si, solution = pickle.load(f)
    return si, solution

def create_schedule(model, sol_file, input_shape=(3,224,224)):
    import torch
    from monet.graph import Graph
    from monet.cvxpy_solver import Solution
    from monet.solver_info import SolverInfo
    import copy
    m = copy.deepcopy(model)
    m.cpu()
    if m._get_name() == 'MobileNetV2':
        m = torch.nn.Sequential(
        m.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=1),
        m.classifier[0], m.classifier[1])
    graph = Graph.create(m, input_shape)
    del m
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    solver_info, solution = load_solution(sol_file)
    schedule = Schedule(graph, solver_info)
    schedule.init_schedule(solution, solver_info.mode)
    return schedule

if __name__ == '__main__':
    import argparse
    import torchvision
    from time import time
    from pathlib import Path
    from monet.cvxpy_solver import Solution

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('bs')
    parser.add_argument('budget')
    parser.add_argument('mode')
    parser.add_argument('solver')
    parser.add_argument(
        "--solution_file", type=str, default="",
        help="If specified, load stored solution file.")
    parser.add_argument(
        "--check_runtime", action="store_true",
        help="Compute the runtime difference between ours and normal model.")
    parser.add_argument(
        "--run_bs", action="store_true",
        help="Run the given batch size.")
    parser.add_argument(
        "--check_diff", action="store_true",
        help="Compute the output (gradient) difference between ours and normal model.")
    parser.add_argument(
        "--ablation", action="store_true",
        help="Do ablation?.")
    args = parser.parse_args()

    budget = float(args.budget)
    import config
    config.budget = budget
    bs = int(args.bs)
    model_name = args.model.split(".")[-1][:-2]
    mode = args.mode
    print("Memory budget ", budget, " GB")
    print("Batch size ", bs)
    print("Model", model_name)
    print("Mode", mode)

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
    if args.check_diff:
        disable_dropout(model)

    if args.check_runtime:
        graph = Graph.create(model, input_shape=(3, height, width))
        model.cuda()
        solvert = -1

        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            input_ = torch.randn((bs, 3, height, width)).cuda()
            solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solver_model = Model(solver_info, budget, args.solver, args.ablation)
            t0 = time()
            solution = solver_model.solve()
            solvert = time() - t0
            del input_
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        input_ = torch.randn((bs, 3, height, width)).cuda()

        schedule = Schedule(graph, solver_info)
        schedule.init_schedule(solution, mode)
        torch.cuda.synchronize()

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
        exit()

    if args.check_diff:
        graph = Graph.create(model, input_shape=(3, height, width))
        model.cuda()
        input_ = torch.randn((bs, 3, height, width)).cuda()
        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solver_model = Model(solver_info, budget, args.solver, args.ablation)
            solution = solver_model.solve()
        schedule = Schedule(graph, solver_info)
        schedule.init_schedule(solution, mode)

        x0 = model(input_)
        if 'googlenet' in args.model:
            (x0[0] + x0[1] + x0[2]).sum().backward()
        else:
            x0.sum().backward()

        x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))

        print('Forward mean absolute difference',
            abs(x0[0] - x1).mean() if 'googlenet' in args.model else abs(x0 - x1).mean())

        schedule.backward(-torch.ones_like(x1))

        print('Gradient of normal model')
        gradient_diff = ["{:.5f} {} {}".format(float(v.grad.mean()), n, v.shape)
               for n, v in model.named_parameters() if v.grad is not None]
        for gd in gradient_diff:
            print(gd)
        exit()

    if args.run_bs:
        graph = Graph.create(model, input_shape=(3, height, width))
        model.cuda()
        solvert = -1
        bs = int(args.bs)
        print("Solver trying batch size %d" % bs)
        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            input_ = torch.randn((bs, 3, height, width)).cuda()
            solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solver_model = Model(solver_info, budget, args.solver, args.ablation)
            t0 = time()
            solution = solver_model.solve()
            solvert = time() - t0
            del input_
        print("Batch size %d feasible" % bs)
        print("Solved in %fs with actual opt taking %fs" % (solvert, solution.solve_time))
        print("Running schedule for batch size %d" % bs)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        input_ = torch.randn((bs, 3, height, width)).cuda()
        schedule = Schedule(graph, solver_info)
        schedule.init_schedule(solution, mode)
        t0 = time()
        x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))
        del input_
        schedule.backward(-torch.ones_like(x1))
        del x1
        torch.cuda.synchronize()
        t1 = time() - t0
        print("Ran schedule for batch %d " % bs)
        torch.cuda.empty_cache()
        mem = torch.cuda.max_memory_allocated() / 2**20
        print("Ran batch %d with peak memory %8.2fM, %fs" % (bs, mem, t1))

        exit()
