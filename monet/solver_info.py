from collections import defaultdict
from functools import lru_cache
import torch
import monet.lm_ops as lm_ops
from monet.meminfo import meminfo, computeinfo
from monet.graph import *
import numpy as np
import sys

MEMORY_MULTIPLIER = 4
# TODO: Remove this and instead use dtype

class SolverNode:
    def __init__(self):
        self.input_shapes = []
        self.output_shapes = []
        self.local_tensors = set() # Includes direct deps + deps due to branching which are kept in memory while recomputing this node
        self.local_memory = -1  # Memory of local tensors
        self.mem = -1   # Memory of output of node
        self.workspace_mem = [] # Extra space over its input and output reqd to compute a node
        self.fixed_mem = -1 # Fixed memory of the input + parameters + increasing memory of param grad
        self.workspace_compute = []
        self.last_used = -1 # Which node uses this node last
        self.recompute_workspace_mem = []
        self.recompute_workspace_compute = []
        self.inplace_workspace_mem = []
        self.inplace_workspace_compute = []

class BwdNode(SolverNode):
    # args.index for bwd node is wrt to solver_info nodes
    def __init__(self,  fwd_node, fwd_id, num_bwd, bwd_op="ip_grad"):
        super().__init__()
        self.fwd_node = fwd_node
        self.fwd_id = fwd_id
        self.bwd_op = bwd_op
        self.stored = []    # Stored parameter gradients
        self.num_bwd = num_bwd
        self.dep_list_fwd = [[] for i in range(self.num_bwd)]
        self.dep_list_bwd = [[] for i in range(self.num_bwd)]
        self.args = [[] for i in range(self.num_bwd)]
        self.has_intermediates = False

    def __repr__(self):
        if self.fwd_node.op == "aten::_convolution" or self.fwd_node.op == "aten::addmm":
            return '<BwdNode %s %s>' % (self.fwd_node, self.bwd_op)
        else:
            return '<BwdNode %s>' % self.fwd_node.op

    def make_args(self):
        self.op = self.fwd_node.op + "_" + self.bwd_op
        for p in range(self.num_bwd):
            self.args[p] = self.dep_list_fwd[p] + self.dep_list_bwd[p]

class FwdNode(SolverNode):
    def __init__(self, node):
        super().__init__()
        self.gnode = node
        self.op = node.op
        self.inbound_nodes = []
        self.args = [[]]
        self.has_intermediates = False
        self.intermediates = [] # Tuple of op_number, solver intermediate node

    # Forward node will have only one set of dependencies
    def make_args(self):
        self.args = [self.inbound_nodes]

    def __repr__(self):
        return '%s' % self.gnode

class IntNode(FwdNode):
    def __init__(self, node, solver_parent_id, op_idx):
        super().__init__(node)
        self.solver_parent_id = solver_parent_id
        self.op_idx = op_idx

    def __repr__(self):
        return 'Int::%s' % self.gnode

class SolverInfo():
    def __init__(self, bs, model_name, mode):
        self.nodes = {}
        self.newnodes = []
        self.loss = 0
        self.edge_list = []
        self.fwd_to_bwd = {}
        self.bwd_to_fwd = {}
        self.solver_to_graph = []
        self.graph_to_solver = {}

        self.bs = bs
        self.model_name = model_name
        self.mode = mode
        self.data_path = "%s_%d_%s" % (self.model_name, self.bs, self.mode)

        self.select_conv_algo = "conv" in self.mode
        self.do_inplace = "inplace" in self.mode
        self.compute_newnode = "newnode" in self.mode

        self.mode = self.mode.replace("inplace_", "") # Remove inplace from the mode after setting do_inplace


    def extract(self, graph: Graph, *args):
        # Create solver graph of forward and backward pass annotated with memory and compute info
        for i, n in enumerate(graph.nodes):
            # Create solver forward nodes
            if isinstance(n, ComputeNode) and n.has_backward:
                self.solver_to_graph.append(i)
                self.graph_to_solver[i] = len(self.solver_to_graph)-1
                snodeid = len(self.nodes)
                self.nodes[snodeid] = FwdNode(n)
                inbound = []
                for (dep, grad) in n.dependencies:
                    if isinstance(graph.nodes[dep], ComputeNode):
                        assert grad == True
                        inbound.append(self.graph_to_solver[dep])
                self.nodes[snodeid].inbound_nodes = inbound

                for idx, op in enumerate(lm_ops.list_ops(self.mode, n.op)):
                    storage = op.backward_storage
                    if not isinstance(storage, list):
                        storage = [storage]
                    for storetype in storage:
                        if isinstance(storetype, lm_ops.IntermediateStorage):
                            # Create intermediate node
                            newnode_op = "int::" + op.__name__
                            ni = ComputeNode([storetype.size(n.shape)], -1, newnode_op, [], False)
                            self.solver_to_graph.append(-1)
                            sintnodeid = len(self.nodes)
                            self.nodes[sintnodeid] = IntNode(ni, snodeid, idx)
                            inbound = []
                            inbound.append(snodeid)
                            self.nodes[sintnodeid].inbound_nodes = inbound
                            self.newnodes.append(sintnodeid)
                            self.nodes[snodeid].has_intermediates = True
                            self.nodes[snodeid].intermediates.append((idx,sintnodeid))

        self.loss = len(self.nodes)
        nloss = ComputeNode(self.nodes[self.loss-1].gnode.shape, -2, "loss::loss", [], False)
        self.nodes[self.loss] = FwdNode(nloss)
        self.nodes[self.loss].inbound_nodes = [self.loss-1]
        self.extract_deps(graph)
        self.get_mem()
        self.get_local_memory()
        self.get_fixed_memory(graph)
        self.get_conv_info()
        self.get_inplace_info()
        self.get_workspace_cost(graph, *args)
        self.size = len(self.nodes)
        # Write solver_info
        import pickle
        si_file = open("../data/si_" + self.data_path + ".pkl", 'wb')
        pickle.dump(self, si_file)
        si_file.close()


    def extract_deps(self, graph):
        # Create solver backward nodes
        for i in sorted(self.nodes, reverse=True):
            if i > self.loss:
                continue
            if self.nodes[i].gnode.has_backward:
                num_bwd = len(lm_ops.list_ops(self.mode, self.nodes[i].op))
                if self.nodes[i].op == "aten::addmm" or (self.nodes[i].op == "aten::_convolution" and not self.nodes[i].gnode.is_depthwise):
                    # First param_grad, then ip_grad
                    # fwd_to_bwd points to ip_grad
                    nbwd1 = BwdNode(self.nodes[i].gnode, i, num_bwd, "param_grad")
                    nbwd2 = BwdNode(self.nodes[i].gnode, i, num_bwd, "ip_grad")
                    self.bwd_to_fwd[len(self.nodes)] = i
                    self.nodes[len(self.nodes)] = nbwd1
                    self.fwd_to_bwd[i] = len(self.nodes)
                    self.bwd_to_fwd[len(self.nodes)] = i
                    self.nodes[len(self.nodes)] = nbwd2
                else:
                    nbwd = BwdNode(self.nodes[i].gnode, i, num_bwd)
                    self.fwd_to_bwd[i] = len(self.nodes)
                    self.bwd_to_fwd[len(self.nodes)] = i
                    self.nodes[len(self.nodes)] = nbwd
        self.fwd_to_bwd[self.loss] = self.loss

        # get deps for backward
        for i in self.nodes:
            if isinstance(self.nodes[i], BwdNode):
                continue
            if self.nodes[i].gnode.has_backward:
                for ni in self.nodes[i].inbound_nodes:
                    num_bwd_prev = len(lm_ops.list_ops(self.mode, self.nodes[ni].op))
                    for p in range(num_bwd_prev):
                        self.nodes[self.fwd_to_bwd[ni]].dep_list_bwd[p].append(self.fwd_to_bwd[i])
                        if self.nodes[ni].op == "aten::addmm" or (self.nodes[ni].op == "aten::_convolution" and not self.nodes[ni].gnode.is_depthwise):
                            # Add bwd deps to param_grad too
                            self.nodes[self.fwd_to_bwd[ni] - 1].dep_list_bwd[p].append(self.fwd_to_bwd[i])

                ops = lm_ops.list_ops(self.mode, self.nodes[i].op)
                num_bwd = len(ops)

                if self.nodes[i].op == "aten::addmm" or (self.nodes[i].op =="aten::_convolution" and not self.nodes[i].gnode.is_depthwise):
                    assert len(self.nodes[i].inbound_nodes) <= 1
                    if len(self.nodes[i].inbound_nodes) == 1:
                        for p in range(num_bwd):
                            self.nodes[self.fwd_to_bwd[i] - 1].dep_list_fwd[p].append(self.nodes[i].inbound_nodes[0])
                else:
                    for p in range(num_bwd):
                        deps = ops[p]().backward_storage
                        l = []
                        if not isinstance(deps, list):
                            deps = [deps]
                        for dep in deps:
                            if isinstance(dep, lm_ops.InputStorage):
                                for inids in dep.ids:
                                    arg_in = self.nodes[i].gnode.args[inids]
                                    if isinstance(arg_in, ComputeNode.D):
                                        innode = graph.nodes[arg_in.index]
                                        if isinstance(innode, ComputeNode):
                                            l.append(self.graph_to_solver[arg_in.index])
                            if isinstance(dep, lm_ops.OutputStorage):
                                l.append(i)
                            if isinstance(dep, lm_ops.IntermediateStorage):
                                added_int = False
                                assert self.nodes[i].has_intermediates
                                for p_option, nint_id in self.nodes[i].intermediates:
                                    nint = self.nodes[nint_id]
                                    assert isinstance(nint, IntNode)
                                    if p_option == p:
                                        l.append(nint_id)
                                        added_int = True
                                assert added_int == True
                        self.nodes[self.fwd_to_bwd[i]].dep_list_fwd[p] = l

            if i == self.loss - 1:  # inject loss node assuming we are at output node
                for p in range(num_bwd):
                    self.nodes[self.fwd_to_bwd[i]].dep_list_fwd[p].append(self.loss)
                    if self.nodes[i].op == "aten::addmm" or (self.nodes[i].op == "aten::_convolution" and not self.nodes[i].gnode.is_depthwise):
                        self.nodes[self.fwd_to_bwd[i] - 1].dep_list_fwd[p].append(self.loss)

        for i in range(self.loss+1):
            self.nodes[i].make_args()
        for i in range(self.loss+1, len(self.nodes)):
            self.nodes[i].make_args()
            fwd_node = self.nodes[i].fwd_node
            stored = []
            output_shapes = []
            # Need to have a list because of nodes like add and cat
            # NOTE for both aten::add and cat, we consider output of bwd as both inputs of fwd
            for (dep, rgrad) in (fwd_node.dependencies):
                nin = graph.nodes[dep]
                if isinstance(nin, Param) and rgrad == True:
                    stored.append(nin)  # Params have gradients which will be stored in backward
                elif isinstance(nin, Param):
                    pass
                elif isinstance(nin, ComputeNode) and rgrad == True:
                    output_shapes.append(list(nin.shape))
                elif isinstance(nin, Input):
                    if rgrad:
                        output_shapes.append(list(nin.shape))
                else:
                    sys.exit("Unknown node encountered ")

            if ((fwd_node.op == "aten::_convolution" and not fwd_node.is_depthwise) or fwd_node.op == "aten::addmm") and self.nodes[i].bwd_op == "ip_grad":
                stored = []
            if ((fwd_node.op == "aten::_convolution" and not fwd_node.is_depthwise) or fwd_node.op == "aten::addmm") and self.nodes[i].bwd_op == "param_grad":
                output_shapes = []
            self.nodes[i].output_shapes = output_shapes
            self.nodes[i].stored = stored

        # Create edge_list
        self.edge_list = []
        for v in self.nodes:
            for k, vdeps in enumerate(self.nodes[v].args):
                for u in vdeps:
                    edge = (u, v, k)
                    self.edge_list.append(edge)


    def get_mem(self):
        for i in self.nodes:
            if isinstance(self.nodes[i], BwdNode):
                self.nodes[i].mem = sum([ np.prod(oshape)* MEMORY_MULTIPLIER * -1 * self.bs if np.prod(oshape)<0 else
                                    np.prod(oshape)* MEMORY_MULTIPLIER for oshape in self.nodes[i].output_shapes])
            else:
                oshape = list(self.nodes[i].gnode.shape)
                self.nodes[i].mem = np.prod(oshape) * MEMORY_MULTIPLIER * -1 * self.bs if np.prod(oshape)<0 else np.prod(oshape)* MEMORY_MULTIPLIER
                if "int::" in self.nodes[i].op:
                    self.nodes[i].mem = int( (self.nodes[i].mem / MEMORY_MULTIPLIER )) # The size of int node already includes memory multipliers according to dtype


    def get_local_memory(self):
        present = -1

        for v in self.nodes:
            # fwd-fwd and bwd-bwd dependencies remain same in all paths
            for u in self.nodes[v].args[0]:
                # dont add forward nodes to backward working set
                if v>self.loss and u<self.loss:
                    continue
                self.nodes[v].local_tensors.add(u)
                self.nodes[u].last_used = v
                if u != present:
                    for ni in range(u+1,v):
                        self.nodes[ni].local_tensors.add(u)
            present = v

        self.nodes[0].local_memory = 0 # Otherwise won't have an entry
        for i in self.nodes:
            s = self.nodes[i].local_tensors
            mem = 0
            for u in s:
                mem = mem + self.nodes[u].mem
            self.nodes[i].local_memory = mem


    def get_fixed_memory(self, graph):
        fixed_mem_fwd = 0
        for i in self.nodes:
            if i >= self.loss:
                break
            for a in self.nodes[i].gnode.args:
                if not isinstance(a, ComputeNode.V) and (isinstance(graph.nodes[a.index], Param) or isinstance(graph.nodes[a.index], Input)):
                    fmem = np.prod(list(graph.nodes[a.index].shape)) * MEMORY_MULTIPLIER
                    if fmem < 0:
                        fmem = fmem * -1 * self.bs
                    fixed_mem_fwd = fixed_mem_fwd + fmem
        self.nodes[0].fixed_mem = fixed_mem_fwd

        for i in self.nodes:
            if i == 0:
                continue
            self.nodes[i].fixed_mem = self.nodes[i-1].fixed_mem
            if i <= self.loss:    # The fixed mem will be counting always
                continue
            for param_stored in self.nodes[i].stored:
                fmem = np.prod(list(param_stored.shape)) * MEMORY_MULTIPLIER
                if fmem < 0:
                    fmem = fmem * -1 * self.bs
                self.nodes[i].fixed_mem = self.nodes[i].fixed_mem + fmem


    def get_workspace_cost(self, graph, *args):
        from pathlib import Path
        import pickle
        workspace_mem = defaultdict(list)
        workspace_compute = defaultdict(list)
        recompute_workspace_mem = defaultdict(list)
        recompute_workspace_compute = defaultdict(list)
        inplace_workspace_mem = defaultdict(list)
        inplace_workspace_compute = defaultdict(list)
        cost = Path("../data/cost_" + self.data_path + ".pkl")
        if cost.is_file():
            pkl_cost = open("../data/cost_" + self.data_path + ".pkl", 'rb')
            workspace_mem, workspace_compute, recompute_workspace_mem, recompute_workspace_compute, inplace_workspace_mem, inplace_workspace_compute = pickle.load(pkl_cost)
            pkl_cost.close()
            assert len(workspace_compute) == len(self.nodes)
            assert len(workspace_mem) == len(self.nodes)
            for i in self.nodes:
                self.nodes[i].workspace_mem = workspace_mem[i]
                self.nodes[i].workspace_compute = workspace_compute[i]
                self.nodes[i].recompute_workspace_mem = recompute_workspace_mem[i]
                self.nodes[i].recompute_workspace_compute = recompute_workspace_compute[i]
                self.nodes[i].inplace_workspace_mem = inplace_workspace_mem[i]
                self.nodes[i].inplace_workspace_compute = inplace_workspace_compute[i]
        # Recomputation memory
        else:
            for i in sorted(self.nodes.keys(), reverse=True):
                n = self.nodes[i]
                if isinstance(n, BwdNode):
                    b = n
                    n = b.fwd_node
                    op_impls = lm_ops.list_ops(self.mode, n.op)
                    for op_impl in op_impls:
                        fwd_working_memory, bwd_working_memory, fwd_working_memory_recompute = meminfo(n, op_impl(), graph, self.bs, b.bwd_op, self.select_conv_algo, self.do_inplace, *args)
                        runtime_fwd, runtime_bwd, runtime_fwd_recompute = computeinfo(n, op_impl(), graph, self.bs, b.bwd_op, self.select_conv_algo, self.do_inplace, *args)
                        print(b, fwd_working_memory, bwd_working_memory, fwd_working_memory_recompute, runtime_fwd, runtime_bwd, runtime_fwd_recompute)
                        if self.select_conv_algo and self.bwd_to_fwd[i] in self.conv_list:
                            for cbwd in range(len(bwd_working_memory)):
                                self.nodes[i].workspace_mem.append(bwd_working_memory[cbwd])
                                self.nodes[i].workspace_compute.append(runtime_bwd[cbwd])
                            if not b.bwd_op == "ip_grad":
                                for cfwd in range(len(fwd_working_memory)):
                                    self.nodes[self.bwd_to_fwd[i]].workspace_mem.append(fwd_working_memory[cfwd])
                                    self.nodes[self.bwd_to_fwd[i]].workspace_compute.append(runtime_fwd[cfwd])
                        elif self.do_inplace and self.bwd_to_fwd[i] in self.inplace_list:
                            self.nodes[i].workspace_mem.append(bwd_working_memory[0])
                            self.nodes[i].workspace_compute.append(runtime_bwd[0])
                            self.nodes[self.bwd_to_fwd[i]].workspace_mem.append(fwd_working_memory[0])
                            self.nodes[self.bwd_to_fwd[i]].workspace_compute.append(runtime_fwd[0])
                            self.nodes[self.bwd_to_fwd[i]].inplace_workspace_mem.append(fwd_working_memory[1])
                            self.nodes[self.bwd_to_fwd[i]].inplace_workspace_compute.append(runtime_fwd[1])
                        else:
                            self.nodes[i].workspace_mem.append(bwd_working_memory[0])
                            self.nodes[i].workspace_compute.append(runtime_bwd[0])
                            if not ( ((n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm") and b.bwd_op == "ip_grad"):
                                if n.op == "aten::batch_norm":
                                    self.nodes[self.bwd_to_fwd[i]].recompute_workspace_mem.append(fwd_working_memory_recompute[0])
                                    self.nodes[self.bwd_to_fwd[i]].recompute_workspace_compute.append(runtime_fwd_recompute[0])
                                self.nodes[self.bwd_to_fwd[i]].workspace_mem.append(fwd_working_memory[0])
                                self.nodes[self.bwd_to_fwd[i]].workspace_compute.append(runtime_fwd[0])
                else:
                    opname = n.op
                    if "int::" in opname:
                        assert isinstance(n, IntNode)
                        parent_node = self.nodes[n.solver_parent_id]
                        p = n.op_idx
                        self.nodes[i].workspace_mem.append(parent_node.workspace_mem[p])
                        self.nodes[i].workspace_compute.append(parent_node.workspace_compute[p] - parent_node.workspace_compute[0])
                        if self.do_inplace and n.solver_parent_id in self.inplace_list:
                            self.nodes[i].inplace_workspace_mem.append(parent_node.inplace_workspace_mem[p])
                            self.nodes[i].inplace_workspace_compute.append(parent_node.inplace_workspace_compute[p] - parent_node.inplace_workspace_compute[0])
                    elif "loss::" in opname:    # For now assuming loss calculation is operation-less
                        self.nodes[i].workspace_compute.append(0)
                        self.nodes[i].workspace_mem.append(0)
            for i in self.nodes:
                if len(self.nodes[i].recompute_workspace_mem) == 0:
                    self.nodes[i].recompute_workspace_mem = self.nodes[i].workspace_mem
                    self.nodes[i].recompute_workspace_compute = self.nodes[i].workspace_compute
                if len(self.nodes[i].inplace_workspace_mem) == 0:
                    self.nodes[i].inplace_workspace_mem = self.nodes[i].workspace_mem
                    self.nodes[i].inplace_workspace_compute = self.nodes[i].workspace_compute
            for i in self.nodes:
                workspace_compute[i] = self.nodes[i].workspace_compute
                workspace_mem[i] = self.nodes[i].workspace_mem
                recompute_workspace_compute[i] = self.nodes[i].recompute_workspace_compute
                recompute_workspace_mem[i] = self.nodes[i].recompute_workspace_mem
                inplace_workspace_compute[i] = self.nodes[i].inplace_workspace_compute
                inplace_workspace_mem[i] = self.nodes[i].inplace_workspace_mem
            cost_file = open("../data/cost_" + self.data_path + ".pkl", 'wb')
            pickle.dump([workspace_mem, workspace_compute, recompute_workspace_mem, recompute_workspace_compute, inplace_workspace_mem, inplace_workspace_compute], cost_file)
            cost_file.close()

    def get_conv_info(self):
        self.conv_list = defaultdict()
        convop = lm_ops.list_ops(self.mode, 'aten::_convolution')[0]
        self.num_conv_algos = 1
        if self.select_conv_algo:
            self.num_conv_algos = max(convop.n_fwd_algos(), convop.n_bwd_ip_algos(), convop.n_bwd_wt_algos())
            # self. = True
            for i in self.nodes:
                if isinstance(self.nodes[i], BwdNode):
                    if "convolution" in self.nodes[i].op and not self.nodes[i].fwd_node.is_depthwise:
                        self.conv_list[i] = len(self.conv_list)
                else:
                    if "convolution" in self.nodes[i].op and not self.nodes[i].gnode.is_depthwise:
                        self.conv_list[i] = len(self.conv_list)

    def get_inplace_info(self):
        self.inplace_list = defaultdict()
        if self.do_inplace:
            for v in range(self.loss):
                if isinstance(self.nodes[v], IntNode):
                    continue
                if hasattr(lm_ops.list_ops(self.mode, self.nodes[v].op)[0],'inplace'):
                    for u in self.nodes[v].args[0]:
                        if self.nodes[u].last_used == v:
                            self.inplace_list[v] = u
                            if self.nodes[v].has_intermediates:
                                for (_, intmd) in self.nodes[v].intermediates:
                                    self.inplace_list[intmd] = u


    # Below is extra info for checkmate solver
    @property
    @lru_cache(maxsize=None)
    def successor_dict(self):
        sucs = defaultdict(list)
        for eidx, (u, v, p) in enumerate(self.edge_list):
            assert p == 0
            sucs[u].append((eidx, v))
        return sucs

    def successors(self, node):
        return {u for (_, u) in self.successor_dict[node]}

    @property
    @lru_cache(maxsize=None)
    def predecessor_dict(self):
        preds = defaultdict(list)
        for eidx, (u, v, p) in enumerate(self.edge_list):
            assert p == 0
            preds[v].append((eidx, u))
        return preds

    def predecessors_indexed(self, node):
        return self.predecessor_dict[node]
