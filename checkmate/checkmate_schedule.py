import torch
from collections import namedtuple
import monet.lm_ops as lm_ops
import numpy as np

from monet.graph import *
from checkmate.checkmate_solver import *
from monet.solver_info import *
from monet.pipelined_solver_info import *
from models.unet import UNet

ScheduleType = namedtuple('ScheduleType', 'recompute store_output delete_nodes store_intermediate')
KEEP_FWDOP = False

class Schedule(Graph):
    def __init__(self, graph: Graph, info: SolverInfo):
        self.si = info
        self._nodes = graph.nodes
        self.lennodes = len(self.nodes)
        self._outputs = graph._outputs
        self._op = []   # List of operations
        self.bs = -1

        # Stored tensors
        self._stored = [None for i in range(self.lennodes)]
        self._stored_intermediate = [None for i in range(self.lennodes)]
        self._bwd_stored = [None for i in range(self.lennodes)]

        # Parameters list
        self._args = []
        self.args_updated = []

        # Preprocessing
        self.computeInstance = []
        for k, n in enumerate(self.nodes):
            if isinstance(n,ComputeNode) and not n.op=="aten::t":
                self.computeInstance.append(k)

    def init_schedule(self, solution: CheckmateSolution, mode):
        T = len(self.si.nodes)
        # Create main structures
        self._op = [None for i in range(self.lennodes)] # List of operations
        self._fwd_schedule = [[] for i in range(T)]     # Forward schedule
        self._bwd_schedule = [[] for i in range(T)]     # Backward schedule
        self.fwdargs = [None for i in range(self.lennodes)] # Index to forward node input tensor
        self.bwdargs = [None for i in range(self.lennodes)] # Index to backward node input tensors

        # Initialize forward pass structures
        for t in range(T):
            for i, n in enumerate(self.nodes):
                if isinstance(n, ComputeNode) and n.op != "aten::t":
                    j = self.si.graph_to_solver[i]
                    ops_list = lm_ops.list_ops(self.si.mode, n.op)
                    if isinstance(self.si, PipelinedSolverInfo) and self.si.nodes[j].has_intermediates:
                        op = ops_list[-1]() # Select intermediate-computing and intermediate-activated operator implementation
                    else:
                        op = ops_list[0]()  # Select the default operator implementations
                    if n.is_depthwise:
                        op.is_depthwise = True
                    s = solution.s[t+1][j] if t<T-1 else False
                    r = solution.r[t][j]
                    f = solution.f[t][j]
                    schedule_intermediate = False
                    storage = op.backward_storage
                    if not isinstance(storage, list):
                        storage = [storage]
                    for store in storage:
                        if isinstance(store, lm_ops.IntermediateStorage):
                            schedule_intermediate = True
                    if r or len(f) or s:
                        self._fwd_schedule[t].append((i,ScheduleType(r, s, f, schedule_intermediate), n.op))
                        self._op[i] = op
                        self.fwdargs[i] = [(a.value,None) if isinstance(a, ComputeNode.V) else (a.index,a.requires_grad) for a in n.args]
                elif isinstance(n, ComputeNode) and n.op == "aten::t":
                    pass
                else:
                    # Node represents a parameter
                    self._fwd_schedule[t].append((i,None,None))
                    self._op[i] = None

        # Initialize backward pass structures
        for k, m in reversed(list(enumerate(self.nodes))):
            # Create backward dependencies
            if isinstance(m, ComputeNode) and m.op != "aten::t":
                j = self.si.fwd_to_bwd[self.si.graph_to_solver[k]]
                n = self.si.nodes[j]
                assert isinstance(n, BwdNode)
                self.bwdargs[k] = {'param':[], 'ip':[]}
                storage_list = self._op[k].backward_storage
                if not isinstance(storage_list, list):
                    storage_list = [storage_list]
                for storage in storage_list:
                    if isinstance(storage, lm_ops.InputStorage):
                        for posi, i in enumerate(storage.ids):
                            idx = m.args[i].index
                            if (((m.op == "aten::_convolution" and not m.is_depthwise) or m.op == "aten::addmm") and n.bwd_op == "ip_grad"):
                                self.bwdargs[k]['param'].append((idx, True, False))
                                if posi == 0:
                                    self.bwdargs[k]['ip'].append((idx,False,False))   # Input tensor for conv/addmm ip grad need not be stored
                                else:
                                    self.bwdargs[k]['ip'].append((idx,True,False))
                            else:
                                self.bwdargs[k]['ip'].append((idx,True,False))
                    elif isinstance(storage, lm_ops.OutputStorage):
                        self.bwdargs[k]['ip'].append((k,True,False))
                    elif isinstance(storage, lm_ops.IntermediateStorage):
                        self.bwdargs[k]['ip'].append((k,True,True))

            # Create backward schedule
            for t in range(T):
                if isinstance(m, ComputeNode) and m.op != "aten::t":
                    j = self.si.fwd_to_bwd[self.si.graph_to_solver[k]]
                    n = self.si.nodes[j]
                    assert isinstance(n, BwdNode)
                    s = solution.s[t+1][j] if t<T-1 else False
                    r = solution.r[t][j]
                    f = solution.f[t][j]
                    if (((m.op == "aten::_convolution" and not m.is_depthwise) or m.op == "aten::addmm") and n.bwd_op == "ip_grad"):
                        s1 = solution.s[t+1][j-1] if t<T-1 else False
                        if solution.r[t][j-1] or len(solution.f[t][j-1]) or s1:
                            self._bwd_schedule[t].append((k,ScheduleType(solution.r[t][j-1], s1, solution.f[t][j-1], False),"param"))
                    if r or len(f) or s:
                        self._bwd_schedule[t].append((k,ScheduleType(r, s, f, False),"ip"))
                elif isinstance(m, ComputeNode) and m.op == "aten::t":
                    pass
                else:
                    self._bwd_schedule[t].append((k,None,"grad"))

        self.opshapes = defaultdict()
        for k in self._outputs:
            self.opshapes[k] = [self.bs if dim==-1 else dim for dim in self._nodes[k].shape]

    def _forward(self, t):
        tensors = self._stored
        bw_tensors = self._bwd_stored
        self._stored = [None] * self.lennodes
        self._bwd_stored = [None] * self.lennodes
        if len(self._fwd_schedule[t]):
            for (k,schedule,op_name) in self._fwd_schedule[t]:
                if schedule == None:
                    tensors[k] = self._args[k]
                else:
                    recompute, s, f, si = schedule
                    if recompute:
                        args = [a if b==None else tensors[a].requires_grad_(b) for (a,b) in self.fwdargs[k]]
                        # Checkmate does not reuse params for BN
                        if op_name == "aten::batch_norm":
                            self._op[k].params = None
                        if si:
                            tensors[k], self._stored_intermediates[k] = self._op[k].forward(*args)
                        else:
                            tensors[k] = self._op[k].forward(*args)
                        assert tensors[k] is not None
                        del args
                    for u in f:
                        assert u < self.si.loss
                        graphu = self.si.solver_to_graph[u]
                        tensors[graphu] = None
                    if s:
                        self._stored[k] = tensors[k]

        if len(self._bwd_schedule[t]):
            for (k,schedule,optype) in self._bwd_schedule[t]:
                if schedule == None:
                    if bw_tensors[k] is not None and k not in self.args_updated and self._args[k].requires_grad:
                        assert len(bw_tensors[k]) == 1
                        for u in bw_tensors[k]:
                            self._args[k].backward(bw_tensors[k][u])
                        bw_tensors[k] = None
                        # self._bwd_stored[k] = None
                        self.args_updated.append(k)
                else:
                    recompute, s, f, si = schedule
                    if recompute:
                        for (idx, checkNone, intmd) in self.bwdargs[k][optype]:
                            if checkNone:
                                if intmd:
                                    assert self._stored_intermediate[idx] is not None
                                else:
                                    assert tensors[idx] is not None
                        stored = [tensors[idx] if not intmd else self._stored_intermediates[idx] for (idx,_,intmd) in self.bwdargs[k][optype]]
                        grad_nd = self._op[k]
                        m = self.nodes[k]
                        if ((m.op == "aten::_convolution" and not m.is_depthwise) or m.op == "aten::addmm"):
                            if optype == "param":
                                grad_nd.algorithm = 0
                            elif optype == "ip":
                                grad_nd.algorithm = 10
                        # Call backward
                        bwd_in = None
                        if k in self._outputs:
                            s = [val if val>0 else self.bs for val in list(self.opshapes[k])]
                            bw_tensors[k] = {-1: torch.ones(s, device=self._args[0].device)}
                        with torch.no_grad():
                            assert bw_tensors[k] is not None, "k, t: %d %d " % (k, t)
                            for u in bw_tensors[k]:
                                assert bw_tensors[k][u] is not None,  "k, u, t: %d %d %s %d " % (k, u, self.si.nodes[self.si.graph_to_solver[k]], t)
                                if bwd_in == None:
                                    bwd_in = bw_tensors[k][u]
                                else:
                                    bwd_in += bw_tensors[k][u]

                            assert bwd_in is not None

                        bw_outs = grad_nd.backward(bwd_in, stored)
                        del bwd_in

                        if not isinstance(bw_outs, (list, tuple)):
                            bw_outs = (bw_outs,)

                        assert len(bw_outs) == len(self.nodes[k].dependencies), \
                            "Require the same number of grad outputs as forward inputs" \
                            " %s (%d) , %s (%d)" % (
                                repr(bw_outs), len(bw_outs),
                                repr(self.nodes[k].dependencies), len(self.nodes[k].dependencies))

                        # Accumulate the backward gradient
                        for (i, r), o in zip(self.nodes[k].dependencies, bw_outs):
                            if r:
                                if o is not None:
                                    if bw_tensors[i] is None:
                                        bw_tensors[i] = {k: o}
                                    else:
                                        bw_tensors[i][k] = o
                        del grad_nd, bw_outs, o

                    for u in f:
                        if u < self.si.loss:
                            graphu = self.si.solver_to_graph[u]
                            tensors[graphu] = None
                        elif u == self.si.loss:
                            pass
                            # Do not delete loss nodes
                        else:
                            graphu = self.si.solver_to_graph[self.si.bwd_to_fwd[u]]
                            unode = self.si.nodes[self.si.bwd_to_fwd[u]].gnode
                            for (i,r) in unode.dependencies:
                                if isinstance(self.nodes[i], ComputeNode):
                                    bw_tensors[i][graphu] = None

                    if s:
                        for (i,r) in self.nodes[k].dependencies:
                            if r:
                                if isinstance(self._nodes[i], ComputeNode):
                                    if optype != "param":
                                        assert bw_tensors[i] is not None
                                        assert bw_tensors[i][k] is not None, "%d (%s) should have bwd input from %d (%s)" % (self.si.graph_to_solver[i], self.si.nodes[self.si.graph_to_solver[i]], self.si.graph_to_solver[k], self.si.nodes[self.si.graph_to_solver[k]])
                                        if self._bwd_stored[i] is None:
                                            self._bwd_stored[i] = {k: bw_tensors[i][k]}
                                        else:
                                            self._bwd_stored[i][k] = bw_tensors[i][k]

        del tensors, bw_tensors

    def forward(self, *args):
        self._args = args
        T =  self.si.size
        fwd_output = None
        for t in range(T):
            self._forward(t)
        for k, n in enumerate(self._nodes):
            if k in self.computeInstance and n.op == "aten::batch_norm":
                self._op[k].params = None
        self.args_updated = []
        return fwd_output

def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

def load_solution(filename):
    import pickle
    print(f'Loading solver_info, solution from {filename}')
    with open(filename, 'rb') as f:
        si, solution = pickle.load(f)
    return si, solution


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
        "--check_diff", action="store_true",
        help="Compute the output (gradient) difference between ours and normal model.")
    parser.add_argument(
        "--check_runtime", action="store_true",
        help="Compute the runtime difference between ours and normal model.")
    parser.add_argument(
        "--run_bs", action="store_true",
        help="Run the given batch size.")
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Pipeline the operator optimization followed by checkpointing")
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

    graph = Graph.create(model, input_shape=(3, height, width))
    model.cuda()

    solvert = -1

    if args.check_diff:
        input_ = torch.randn((bs, 3, height, width)).cuda()
        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solution = solve_ilp_gurobi(solver_info, budget, approx=False, time_limit=86400)
        schedule = Schedule(graph, solver_info)
        schedule.init_schedule(solution, mode)

        x0 = model(input_)
        if 'googlenet' in args.model:
            (-(x0[0]+x0[1]+x0[2])).sum().backward()
        else:
            (-x0).sum().backward()
        KEEP_FWDOP = True
        x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))

        print('Forward mean absolute difference',
            abs(x0[0] - x1).mean() if 'googlenet' in args.model else abs(x0 - x1).mean())
        print('original output', x0)
        print('ours output', x1)

        print('Gradient of normal model',
              ['{:.5f} {}'.format(float(v.grad.mean()), v.shape)
               for v in model.parameters() if v.grad is not None])

    if args.check_runtime:
        FORWARD_EMPTY_CACHE = False

        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            input_ = torch.randn((bs, 3, height, width)).cuda()
            if args.pipeline:
                solver_info = PipelinedSolverInfo(bs=bs, model_name=model_name, mode=mode)
            else:
                solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solution = solve_ilp_gurobi(solver_info, budget, approx=False, time_limit=86400)
            # t0 = time()
            # solution = solver_model.solve()
            # solvert = time() - t0
            del input_
        input_ = torch.randn((bs, 3, height, width)).cuda()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        schedule = Schedule(graph, solver_info)
        schedule.bs = bs
        schedule.init_schedule(solution, mode)
        torch.cuda.synchronize()

        start_event_monet = torch.cuda.Event(enable_timing=True)
        end_event_monet = torch.cuda.Event(enable_timing=True)
        for iterid in range(120):
            if iterid == 100:
                start_event_monet.record()
            x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))
            for v in model.parameters():
                v.grad = None
        end_event_monet.record()
        torch.cuda.synchronize()
        del x1
        autosave_maxmem = torch.cuda.max_memory_allocated() / 2**20

        print("checkmate: %f ms avg, %8.2f MB" % (start_event_monet.elapsed_time(end_event_monet)/20, autosave_maxmem))

        exit()

    solvert = -1
    if args.run_bs:
        bs = int(args.bs)
        print("Solver trying batch size %d" % bs)
        if len(args.solution_file) > 0:
            solver_info, solution = load_solution(args.solution_file)
        else:
            input_ = torch.randn((bs, 3, height, width)).cuda()
            if args.pipeline:
                solver_info = PipelinedSolverInfo(bs=bs, model_name=model_name, mode=mode)
            else:
                solver_info = SolverInfo(bs=bs, model_name=model_name, mode=mode)
            solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
            solution = solve_ilp_gurobi(solver_info, budget, approx=False, time_limit=86400)
            del input_
        print("Batch size %d feasible" % bs)
        print("Solved in %fs with actual opt taking %fs" % (solvert, solution.solve_time))
        print("Running schedule for batch size %d" % bs)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        input_ = torch.randn((bs, 3, height, width)).cuda()
        schedule = Schedule(graph, solver_info)
        schedule.bs = bs
        schedule.init_schedule(solution, mode)
        t0 = time()
        x1 = schedule.forward(input_, *list(model.state_dict(keep_vars=True).values()))
        del input_
        del x1

        torch.cuda.synchronize()
        t1 = time() - t0
        print("Ran schedule for batch %d " % bs)
        torch.cuda.empty_cache()
        mem = torch.cuda.max_memory_allocated() / 2**20
        print("Ran batch %d with peak memory %8.2fM, %fs" % (bs, mem, t1))
