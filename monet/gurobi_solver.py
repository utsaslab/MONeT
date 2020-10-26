from enum import Enum
import logging
import math
import os
from typing import Dict, Any

import numpy as np
from gurobipy import GRB, Model, quicksum


from monet.graph import *
from monet.solver_info import *

from models.unet import UNet

CPU_GCD_MULTIPLIER = 100
MEM_GCD_MULTIPLIER = 1
MEMORY_MULTIPLIER = 4
GB_TO_KB = 1024*1024 #for budget

class Solution:
    def __init__(self, r, s, m, rf, ip, p, solve_time, fwd_init, bwd_init, runtime_expected):
        self.r = r
        self.s = s
        self.m = m
        self.rf = rf
        self.ip = ip
        self.p = p
        self.solve_time = solve_time
        self.fwd_init = fwd_init
        self.bwd_init = bwd_init
        self.runtime_expected = runtime_expected

class ILPSolver:
    def __init__(self, si: SolverInfo, budget, gurobi_params: Dict[str, Any] = None, ablation=False, overhead=False):
        self.gurobi_params = gurobi_params
        self.num_threads = self.gurobi_params.get('Threads', 1)
        self.budget = int(budget * MEM_GCD_MULTIPLIER * GB_TO_KB)
        self.si: SolverInfo = si
        self.solve_time = None
        self.ablation = ablation
        self.overhead = overhead

        V = self.si.loss + 1
        T = len(self.si.nodes) - self.si.loss
        Y = 3
        budget = self.budget

        self.m = Model("monet{}".format(self.budget))
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        self.ram = np.array([math.ceil(self.si.nodes[i].mem*MEM_GCD_MULTIPLIER/1024) for i in self.si.nodes]) # Convert to KB
        self.cpu = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].workspace_compute] ) for i in self.si.nodes)
        self.cpu_recompute = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].recompute_workspace_compute] ) for i in self.si.nodes)
        self.cpu_inplace = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].inplace_workspace_compute] ) for i in self.si.nodes)

        self.R = self.m.addVars(T, V, name="R", vtype=GRB.BINARY)   # Recomputation
        self.P = self.m.addVars(T, V, name="P", vtype=GRB.BINARY)   # In-memory
        self.S = self.m.addVars(T+1, V, name="S", vtype=GRB.BINARY) # Stored
        self.M = self.m.addVars(T, Y, name="M", vtype=GRB.BINARY)   # Backward operator implementation
        self.SM = self.m.addVars(T, V, Y, name="SM", vtype=GRB.BINARY)  # Linearization of S * M
        if self.si.select_conv_algo:
            self.RF = self.m.addVars(T, len(self.si.conv_list), self.si.num_conv_algos, name="RF", vtype=GRB.BINARY) # Conv operator implementations
        if self.si.do_inplace:
            self.IP = self.m.addVars(T, V, name="IP", vtype=GRB.BINARY) # In-place

    def cmem(self, j, recompute=False, inplace=False):
        if inplace:
            return [math.ceil(val * MEM_GCD_MULTIPLIER * 1024) for val in self.si.nodes[j].inplace_workspace_mem]
        else:
            if recompute:
                return [math.ceil(val * MEM_GCD_MULTIPLIER * 1024) for val in self.si.nodes[j].recompute_workspace_mem]
            else:
                return [math.ceil(val * MEM_GCD_MULTIPLIER * 1024) for val in self.si.nodes[j].workspace_mem]

    def local_memory(self, j):
        return math.ceil(self.si.nodes[j].local_memory * MEM_GCD_MULTIPLIER / 1024)

    def fixed_ram(self, j):
        return math.ceil(self.si.nodes[j].fixed_mem * MEM_GCD_MULTIPLIER / 1024)

    def build_model(self):
        V = self.si.loss + 1
        T = len(self.si.nodes) - self.si.loss
        Y = 3
        budget = self.budget

        # define objective function
        if self.si.select_conv_algo:
            fwd_compute = quicksum(self.R[0, i] * self.cpu[i][0] for i in range(V) if i not in self.si.conv_list)
            fwd_recompute = quicksum(quicksum(self.R[t, i]*self.cpu_recompute[i][0] for t in range(1,T)) for i in range(V) if i not in self.si.conv_list)
            conv_fwd_compute = quicksum(quicksum(self.RF[t,self.si.conv_list[i],c] * self.cpu[i][c] for t in range(T)) for i in self.si.conv_list if i<=self.si.loss for c in range(len(self.cpu[i]))) # conv's compute and recompute same
            if self.si.do_inplace:
                inplace_fwd_compute = quicksum(quicksum(self.IP[t,i] * (self.cpu_inplace[i][0]-self.cpu[i][0]) for t in range(T)) for i in self.si.inplace_list) # relu's compute and recompute same
            conv_bwd_compute = quicksum(quicksum(self.RF[t,self.si.conv_list[i],c] * self.cpu[i][c] for t in range(T)) for i in self.si.conv_list if i>self.si.loss for c in range(len(self.cpu[i]))) # conv's compute and recompute same
            bwd_compute = quicksum(self.M[t, p] * self.cpu[t+V-1][p] for t in range(T) for p in range(Y) if (p<len(self.cpu[t+V-1]) and (t+V-1) not in self.si.conv_list))
            if self.si.do_inplace:
                compute_fwd = fwd_compute + fwd_recompute + conv_fwd_compute + inplace_fwd_compute
            else:
                compute_fwd = fwd_compute + fwd_recompute + conv_fwd_compute
            compute_bwd = conv_bwd_compute + bwd_compute
            compute_cost = compute_fwd + compute_bwd
        else:
            fwd_compute = quicksum(self.R[0, i] * self.cpu[i][0] for i in range(V))
            fwd_recompute = quicksum(self.R[t, i] * self.cpu_recompute[i][0] for t in range(1,T) for i in range(V))
            if self.si.do_inplace:
                inplace_fwd_compute = quicksum(quicksum(self.IP[t,i] * (self.cpu_inplace[i][0]-self.cpu[i][0]) for t in range(T)) for i in self.si.inplace_list) # relu's compute and recompute same
            bwd_compute = quicksum(self.M[t, p] * self.cpu[t+V-1][p] for t in range(T) for p in range(Y) if p<len(self.cpu[t+V-1]))
            if self.si.do_inplace:
                compute_fwd = fwd_compute + fwd_recompute + inplace_fwd_compute
            else:
                compute_fwd = fwd_compute + fwd_recompute
            compute_bwd = bwd_compute
            compute_cost = compute_fwd + compute_bwd
        self.m.setObjective(compute_cost, GRB.MINIMIZE)
        self.compute_cost = compute_cost
        self.compute_fwd = compute_fwd
        self.compute_bwd = compute_bwd

        # Add in-place constraints
        for t in range(T):
            for v in range(V):
                if self.si.do_inplace:
                    self.m.addLConstr(self.P[t,v], GRB.LESS_EQUAL, self.R[t,v])
                    self.m.addLConstr(self.IP[t,v], GRB.LESS_EQUAL, self.R[t,v])
                    # print(list(self.si.inplace_list.keys()))
                    if v not in self.si.inplace_list:
                        self.m.addLConstr(self.IP[t,v], GRB.EQUAL, 0)
                    elif not isinstance(self.si.nodes[v], IntNode):
                        # If IP[v], then P[u] = 0, else P[u] = R[u]
                        self.m.addLConstr(self.P[t,self.si.inplace_list[v]], GRB.GREATER_EQUAL, self.R[t,self.si.inplace_list[v]] - 2*self.IP[t,v])
                        self.m.addLConstr(self.P[t,self.si.inplace_list[v]], GRB.LESS_EQUAL, 2 - 2*self.IP[t,v])
                        self.m.addLConstr(self.S[t+1,self.si.inplace_list[v]], GRB.LESS_EQUAL, 2 - 2*self.IP[t,v])
                else:
                    self.m.addLConstr(self.P[t,v], GRB.EQUAL, self.R[t,v])
        # store nothing in the beginning
        for i in range(V):
            self.m.addLConstr(self.S[0,i] , GRB.EQUAL, 0)
        # Recompute full forward
        for i in range(V):
            if not isinstance(self.si.nodes[i], IntNode):
                self.m.addLConstr(self.R[0,i] , GRB.EQUAL, 1)
        # All M which have no paths are set as 0
        self.m.addLConstr(self.M[0,0], GRB.EQUAL, 1)
        for p in range(1,Y):
            self.m.addLConstr(self.M[0,p], GRB.EQUAL, 0)
        for t in range(1, T):
            bkwd_t = V + t - 1
            paths = len(self.si.nodes[bkwd_t].args)
            for p in range(paths, Y):
                self.m.addLConstr(self.M[t,p], GRB.EQUAL, 0)
        # Set failed conv's RF to be 0
        if self.si.select_conv_algo:
            for i in self.si.conv_list:
                for c in range(self.si.num_conv_algos):
                    if c >= len(self.cpu[i]) or self.cpu[i][c] < 0:
                        for t in range(T):
                            self.m.addLConstr(self.RF[t,self.si.conv_list[i],c], GRB.EQUAL, 0)

        # create constraints for boolean multiplication linearization
        for p in range(Y):
            for i in range(V):
                for t in range(T):
                    self.m.addLConstr(self.SM[t,i, p], GRB.GREATER_EQUAL, self.M[t,p] + self.S[t+1,i] - 1)
                    self.m.addLConstr(self.SM[t,i, p], GRB.LESS_EQUAL, self.M[t,p])
                    self.m.addLConstr(self.SM[t,i, p], GRB.LESS_EQUAL, self.S[t+1,i])

        if self.ablation:
            print("Doing ablation")
            # Disable all recomputation
            for t in range(1,T):
                for v in range(V):
                    self.m.addLConstr(self.R[t,v], GRB.EQUAL, 0)
            # Fix all operations to be inplace
            for t in range(T):
                for v in range(V):
                    if self.si.do_inplace and v in self.si.inplace_list:
                        self.m.addLConstr(self.IP[t,v], GRB.EQUAL, self.R[t,v])

        # Correctness constraints
        # Ensure all checkpoints are in memory
        for t in range(T):
            for i in range(V):
                self.m.addLConstr(self.S[t + 1, i], GRB.LESS_EQUAL, self.S[t, i] + self.P[t, i])
        # At least one path should be used
        for t in range(T):
            self.m.addLConstr(quicksum(self.M[t,p] for p in range(Y)), GRB.GREATER_EQUAL, 1)
        # Ensure all computations are possible
        for (u, v, p) in self.si.edge_list:
            if u < V and v <V:
                for t in range(T):
                    self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.R[t, u] + self.S[t, u])
            if u < V and v >= V:
                t = v + 1 - V
                self.m.addLConstr(self.M[t,p], GRB.LESS_EQUAL, self.P[t, u] + self.S[t+1, u])
        # Ensure that new nodes only computed if main node computed
        for t in range(T):
            for i in range(V):
                if self.si.nodes[i].has_intermediates:
                    for (_,intid) in self.si.nodes[i].intermediates:
                        self.m.addLConstr(self.R[t, intid], GRB.LESS_EQUAL, self.R[t,i])
                        if self.si.do_inplace and i in self.si.inplace_list:
                            self.m.addLConstr(self.IP[:,intid], GRB.GREATER_EQUAL, self.IP[:,i] + self.R[:,intid] -1)
                            self.m.addLConstr(self.IP[:,intid], GRB.LESS_EQUAL, self.IP[:,i])

        # Constraints for conv selection
        if self.si.select_conv_algo:
            for i in self.si.conv_list:
                if i < self.si.loss:
                    for t in range(T):
                        self.m.addLConstr(quicksum(self.RF[t,self.si.conv_list[i],c] for c in range(self.si.num_conv_algos)), GRB.GREATER_EQUAL, self.R[t,i])
                else:
                    t_bwdi = i + 1 - V
                    self.m.addLConstr(quicksum(self.RF[t_bwdi,self.si.conv_list[i],c] for c in range(self.si.num_conv_algos)), GRB.GREATER_EQUAL, 1)
                    for c in range(self.si.num_conv_algos):
                        for t in range(t_bwdi):
                            self.m.addLConstr(self.RF[t,self.si.conv_list[i],c], GRB.EQUAL, 0)
                        for t in range(t_bwdi+1, T):
                            self.m.addLConstr(self.RF[t,self.si.conv_list[i],c], GRB.EQUAL, 0)

        # Memory constraints
        for t in range(T):
            bkwd_t = V + t - 1
            bwd_node = self.si.nodes[bkwd_t]
        for t in range(T):
            bkwd_t = V + t - 1
            bwd_node = self.si.nodes[bkwd_t]
            bwd_local_memory = self.local_memory(bkwd_t) if t!=0 else 0
            gradm = bwd_local_memory + self.fixed_ram(bkwd_t-1)
            for j in range(V):
                keep_tensors = [[fwdin for fwdin in bwd_node.args[p] if fwdin <= self.si.loss] for p in range(len(bwd_node.args))]
                # Forward checkpoint constraint
                self.m.addLConstr(  gradm +
                                    quicksum(self.S[t,i]*self.ram[i] for i in range(j, V)) +
                                    quicksum(self.S[t+1,i]*self.ram[i] for i in range(j) if t<T-1) +
                                    quicksum( quicksum( self.M[t,p]*self.ram[i] for i in range(j) if i in keep_tensors[p]) for p in range(len(bwd_node.args)) ) -
                                    quicksum( quicksum( self.SM[t,i, p]*self.ram[i] for i in range(j) if i in keep_tensors[p]) for p in range(len(bwd_node.args)) ),
                                    GRB.LESS_EQUAL, budget)

                workspace_mem = self.cmem(j) if t==0 else self.cmem(j, recompute=True, inplace=False)
                if self.si.select_conv_algo and j in self.si.conv_list:
                    wm = quicksum(self.RF[t,self.si.conv_list[j],c] * workspace_mem[c] for c in range(len(workspace_mem)))
                elif self.si.do_inplace and j in self.si.inplace_list:
                    inplace_workspace_mem = self.cmem(j, recompute=False, inplace=True)
                    wm = self.IP[t,j]*(inplace_workspace_mem[0] - workspace_mem[0]) + self.R[t,j] * workspace_mem[0]
                else:
                    wm = self.R[t,j] * workspace_mem[0]
                # Forward recomputation constraint
                self.m.addLConstr(  gradm + wm + self.R[t,j]*self.ram[j] + self.R[t,j]*self.local_memory(j) +
                                    quicksum(self.S[t,i]*self.ram[i] for i in range(j+1, V)) +
                                    quicksum(self.S[t+ 1,i]*self.ram[i] for i in range(j) if i not in self.si.nodes[j].local_tensors) +
                                    quicksum( quicksum( self.M[t,p]*self.ram[i] for i in range(j) if i in keep_tensors[p]) for p in range(len(bwd_node.args)) ) -
                                    quicksum( quicksum( self.SM[t,i, p]*self.ram[i] for i in range(j) if i in keep_tensors[p]) for p in range(len(bwd_node.args)) ),
                                    GRB.LESS_EQUAL, budget)
                if t>0:
                    if self.si.select_conv_algo and bkwd_t in self.si.conv_list:
                        wmem = self.cmem(bkwd_t)
                        bwd_wm = quicksum(self.RF[t,self.si.conv_list[bkwd_t],c] * wmem[c] for c in range(len(wmem)))
                    else:
                        wmem = self.cmem(bkwd_t)
                        bwd_wm = quicksum(self.M[t,p]*wmem[p] for p in range(len(bwd_node.args)))
                    rammem = [sum([self.ram[i] for i in bwd_node.args[p] if i<V]) for p in range(len(bwd_node.args))]
                    # Backward constraint
                    self.m.addLConstr(  bwd_wm +
                                        bwd_local_memory + self.fixed_ram(bkwd_t) + self.ram[bkwd_t] +
                                        quicksum(self.M[t,p]*rammem[p] for p in range(len(bwd_node.args))) +
                                        quicksum(quicksum(self.SM[t,i,p]*self.ram[i]  for i in range(V) if i not in bwd_node.args[p]) for p in range(len(bwd_node.args))),
                                        GRB.LESS_EQUAL, budget)
        return None

    def solve(self):
        from time import time
        V = self.si.loss + 1
        T = len(self.si.nodes) - self.si.loss
        Y = 3
        budget = self.budget
        self.m.Params.TimeLimit = self.time_limit
        t0 = time()
        self.m.optimize()
        self.solve_time = time() - t0

        infeasible = (self.m.status == GRB.INFEASIBLE)
        if infeasible:
            self.m.computeIIS()
            self.m.write("model.ilp")
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")

        if self.m.solCount < 1:
            raise ValueError(f"Model status is {self.m.status} (not infeasible), but solCount is {self.m.solCount}")

        Rout = np.zeros((T, V), dtype=bool)
        Pout = np.zeros((T, V), dtype=bool)
        Sout = np.zeros((T+1, V), dtype=bool)
        Mout = np.zeros((T, Y), dtype=bool)
        SMout = np.zeros((T, V, Y), dtype=bool)
        RFout = np.zeros((T, len(self.si.conv_list), self.si.num_conv_algos), dtype=bool)
        IPout = np.zeros((T, V), dtype=bool)

        try:
            for t in range(T):
                for i in range(V):
                    Rout[t][i] = round(self.R[t, i].X)
                    Pout[t][i] = round(self.P[t, i].X)
                    Sout[t][i] = round(self.S[t, i].X)
                    if self.si.do_inplace:
                        IPout[t][i] = round(self.IP[t, i].X)
                    for p in range(Y):
                        SMout[t][i][p] = round(self.SM[t, i, p].X)
                for p in range(Y):
                    Mout[t][p] = round(self.M[t, p].X)
                if self.si.select_conv_algo:
                    for i in range(len(self.si.conv_list)):
                        for c in range(self.si.num_conv_algos):
                            RFout[t][i][c] = round(self.RF[t,i,c].X)
            for i in range(V):
                Sout[T][i] = round(self.S[T, i].X)

        except AttributeError as e:
            logging.exception(e)
            return None, None, None, None, None, None, None

        solution = Solution(Rout, Sout, Mout, RFout, IPout, Pout, self.solve_time, -1, -1, -1)
        return solution

def solve_ilp_gurobi(si: SolverInfo, budget: float, time_limit = None, ablation=False):
    """
    Memory-accurate solver with garbage collection.
    :param si: SolverInfo -- graph definition extracted from model
    :param budget: int -- budget constraint for solving
    :param time_limit: int -- time limit for solving in seconds
    """
    param_dict = {'LogToConsole': 1 if print_to_console else 0,
                  'LogFile': str(write_log_file) if write_log_file is not None else "",
                  'Threads': min(os.cpu_count(), 12),
                  'TimeLimit': time_limit,
                  'Method': 1}
    ilpsolver = ILPSolver(si, budget, gurobi_params=param_dict, ablation=ablation, overhead=overhead)
    ilpsolver.build_model()

    try:
        solution = ilpsolver.solve()
        ilp_feasible = True
    except ValueError as e:
        logging.exception(e)
        solution = Solution(None, None, None, None, -1,-1,-1,-1)
        ilp_feasible = False

    fname = '../data/gurobi_solution_%s_%d_%s_%4.2f' % (si.model_name, si.bs, si.mode, budget)
    if ilpsolver.ablation:
        fname = fname+"_ablation"
    fname = fname+".pkl"
    if ilp_feasible:
        import pickle
        solfile = open(fname, 'wb')
        pickle.dump([si, solution], solfile)
        solfile.close()

    print('solvetime(s) %f runtime_expected(s) %.3f fwd_init(s) %.3f bwd_init(s) %.3f' % (solution.solve_time, solution.runtime_expected/CPU_GCD_MULTIPLIER/1000, solution.fwd_init/CPU_GCD_MULTIPLIER/1000, solution.bwd_init/CPU_GCD_MULTIPLIER/1000) )
    return solution

if __name__ == '__main__':
    import argparse
    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('bs')
    parser.add_argument('budget')
    parser.add_argument('mode')
    parser.add_argument(
    "--ablation", action="store_true",
    help="Disable checkpointing.")
    parser.add_argument(
    "--time_limit", default=86400, type=int,
    help="Solver time limit.")
    args = parser.parse_args()

    budget = float(args.budget)
    bs = int(args.bs)
    model_name = args.model.split(".")[-1][:-2]
    mode = args.mode
    time_limit = args.time_limit
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

    graph = Graph.create(model, input_shape=(3, height, width))
    model.cuda()
    input_ = torch.randn([bs,3,height,width]).cuda()
    solver_info = SolverInfo(bs, mode=mode, model_name=model_name)
    solver_info.extract(graph, input_, *list(model.state_dict(keep_vars=True).values()))
    solution = solve_ilp_gurobi(solver_info, budget, time_limit=time_limit, ablation=args.ablation)
