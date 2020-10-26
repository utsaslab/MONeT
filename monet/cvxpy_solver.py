import cvxpy as cp
import numpy as np
import math

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

class Model:
    def __init__(self, si: SolverInfo, budget, time_limit=86400, solver="GUROBI", ablation=False):
        self.si = si
        self.budget = budget
        self.time_limit = time_limit
        self.solver = solver
        self.ablation = ablation
        self.ram = np.array([math.ceil(self.si.nodes[i].mem*MEM_GCD_MULTIPLIER/1024) for i in self.si.nodes]) # Convert to KB
        self.cpu = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].workspace_compute] ) for i in self.si.nodes)
        self.cpu_recompute = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].recompute_workspace_compute] ) for i in self.si.nodes)
        self.cpu_inplace = dict(( i, [math.ceil(val*CPU_GCD_MULTIPLIER) for val in self.si.nodes[i].inplace_workspace_compute] ) for i in self.si.nodes)

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

    def solve(self):
        constraints = []
        budget = self.budget*MEM_GCD_MULTIPLIER*GB_TO_KB
        V = self.si.loss + 1
        T = len(self.si.nodes) - self.si.loss
        Y = 3
        # Init variables
        R = cp.Variable((T,V), name='R', integer=True)  # Recomputation
        P = cp.Variable((T,V), name='P', integer=True)  # Present in-memory
        S = cp.Variable((T+1,V), name='S', integer=True)    # Stored
        M = cp.Variable((T,Y), name='M', integer=True)  # Backward operator implementation
        SM = {} # Linearization of S * M
        for p in range(Y):
            SM[p] = cp.Variable((T,V), name="SM", integer=True)

        RF = {} # Conv operator implementations
        if self.si.select_conv_algo:
            for c in range(self.si.num_conv_algos):
                RF[c] = cp.Variable((T,len(self.si.conv_list)), name='RF', integer=True)

        if self.si.do_inplace:
            IP = cp.Variable((T,V), name='IP', integer=True)    # In-place selection

        constraints.append(0 <= R)
        constraints.append(R <= 1)
        constraints.append(0 <= P)
        constraints.append(P <= 1)
        constraints.append(0 <= S)
        constraints.append(S <= 1)
        constraints.append(0 <= M)
        constraints.append(M <= 1)

        for p in range(Y):
            constraints.append(0 <= SM[p])
            constraints.append(SM[p] <= 1)

        if len(RF):
            for calgo in range(self.si.num_conv_algos):
                constraints.append(0 <= RF[calgo])
                constraints.append(RF[calgo] <= 1)

        # Add in-place constraints
        if self.si.do_inplace:
            constraints.append(0 <= IP)
            constraints.append(IP <= 1)
            constraints.append(P <= R)
            constraints.append(IP <= R)
            not_inplace = np.ones(V)
            not_inplace[list(self.si.inplace_list.keys())] = 0
            not_inplace = np.repeat([not_inplace], T, axis=0)
            print(list(self.si.inplace_list.keys()))
            for v in range(V):
                if v not in self.si.inplace_list:
                    constraints.append(IP[:,v] == 0)
                elif not isinstance(self.si.nodes[v], IntNode):
                    # If IP[v], then P[u] = 0, else P[u] = R[u]
                    constraints.append(P[:,self.si.inplace_list[v]] >= R[:,self.si.inplace_list[v]] - 2*IP[:,v])
                    constraints.append(P[:,self.si.inplace_list[v]] <= 2 - 2*IP[:,v])
                    constraints.append(S[1:,self.si.inplace_list[v]] <= 2 - 2*IP[:,v])
        else:
            constraints.append(P == R)

        # Store nothing in the beginning
        constraints.append(S[0,:] == 0)

        # Recompute full forward
        for i in range(V):
            if not isinstance(self.si.nodes[i], IntNode):
                constraints.append(R[0,i] == 1)

        # All M which have no paths are set as 0
        constraints.append(M[0,0] == 1)
        constraints.append(M[0,1:Y] == 0)
        for t in range(1,T):
            bkwd_t = V + t -1
            paths = len(self.si.nodes[bkwd_t].args)
            if paths < Y:
                constraints.append(M[t,paths:Y] == 0)

        # Set failed conv's RF to be 0
        if len(RF):
            for i in self.si.conv_list:
                for c in range(self.si.num_conv_algos):
                    if c >= len(self.cpu[i]) or self.cpu[i][c] < 0:
                        constraints.append(RF[c][:,self.si.conv_list[i]] == 0)

        for p in range(Y):
            for i in range(V):
                constraints.append(SM[p][:,i]  >= M[:,p] + S[1:,i] - 1)
                constraints.append(SM[p][:,i] <= M[:,p])
                constraints.append(SM[p][:,i] <= S[1:,i])

        # Ablation constraint
        if self.ablation:
            constraints.append(R[1:T,:] == 0)

            if self.si.do_inplace:
                for v in range(V):
                    if v in self.si.inplace_list:
                        constraints.append(IP[:,v] == R[:,v])
        # Correctness constraints
        # Ensure all checkpoints are in memory
        constraints.append(S[1:T+1,:] <= S[:T,:] + P[:T,:])

        # At least one path should be used
        constraints.append(cp.sum(M, axis=1) >= 1)

        # Ensure all computations are possible
        for (u, v, p) in self.si.edge_list:
            if u < V and v <V:
                constraints.append(R[:,v] <= R[:,u] + S[:T,u])
            if u < V and v >= V:
                t = v + 1 - V
                constraints.append(M[t,p] <= P[t,u] + S[t+1,u])

        # Ensure that newnodes are only computed if parent node is computed
        for i in range(V):
            if self.si.nodes[i].has_intermediates:
                for (_,intid) in self.si.nodes[i].intermediates:
                    constraints.append(R[:,intid] <= R[:,i])
                    if self.si.do_inplace and i in self.si.inplace_list:
                        constraints.append(IP[:,intid] >= IP[:,i] + R[:,intid] -1)
                        constraints.append(IP[:,intid] <= IP[:,i])

        # Constraints for conv selection
        if len(RF):
            for i in self.si.conv_list:
                if i < self.si.loss:
                    constraints.append(cp.sum([RF[c][:,self.si.conv_list[i]] for c in range(self.si.num_conv_algos)]) >= R[:,i])
                else:
                    t_bwdi = i + 1 - V
                    constraints.append(cp.sum([RF[c][t_bwdi,self.si.conv_list[i]] for c in range(self.si.num_conv_algos)]) >= 1)
                    for c in range(self.si.num_conv_algos):
                        constraints.append(RF[c][:t_bwdi,self.si.conv_list[i]] == 0)
                        if t_bwdi+1 < T:
                            constraints.append(RF[c][t_bwdi+1:,self.si.conv_list[i]] == 0)

        # Memory constraints
        gradm = np.zeros((T,1))
        ram_keep_tensors = [np.zeros((T,V)) for p in range(Y)]
        ones_tbyv = np.ones((T,V))
        for t in range(T):
            bkwd_t = V + t - 1
            bwd_node = self.si.nodes[bkwd_t]
            bwd_local_memory = self.local_memory(bkwd_t) if t!=0 else 0
            gradm[t] = bwd_local_memory + self.fixed_ram(bkwd_t-1)
        for j in range(V):
            if j == 0:
                sm1 = S[:T, j:V] @ self.ram[:V].reshape(V,1)[j:V]
            else:
                sm1 = (S[:T, j:V] @ self.ram[:V].reshape(V,1)[j:V]) + (S[1:, :j] @ self.ram[:V].reshape(V,1)[:j])

            workspace_mem = np.repeat([self.cmem(j, recompute=True, inplace=False)], T, axis=0)
            workspace_mem[0] = self.cmem(j)
            if len(RF) and j in self.si.conv_list:
                wm = sum([cp.multiply(RF[c][:,None,self.si.conv_list[j]], workspace_mem[:,None,c]) for c in range(workspace_mem.shape[1])])
            elif self.si.do_inplace and j in self.si.inplace_list:
                inplace_workspace_mem = np.repeat([self.cmem(j, recompute=False, inplace=True)], T, axis=0)
                wm = cp.multiply(IP[:,None,j], (inplace_workspace_mem[:,None,0] - workspace_mem[:,None,0])) + cp.multiply(R[:,None,j],workspace_mem[:,None,0])
            else:
                wm = cp.multiply(R[:,None,j],workspace_mem[:,None,0])
            local_mem = self.local_memory(j)
            rm = R[:,None,j]*self.ram[j] + R[:,None,j]*local_mem
            sm2_ram = self.ram.copy()
            if len(list(self.si.nodes[j].local_tensors)):
                sm2_ram[list(self.si.nodes[j].local_tensors)] = 0
            if j == 0:
                sm2 = S[:T,j+1:V] @ sm2_ram[:V].reshape(V,1)[j+1:V]
            elif j+1<V:
                sm2 = S[:T,j+1:V] @ sm2_ram[:V].reshape(V,1)[j+1:V] + S[1:,:j] @ sm2_ram[:V].reshape(V,1)[:j]
            else:
                sm2 = S[1:,:j] @ sm2_ram[:V].reshape(V,1)[:j]

            # for p in range(Y):
            lm1 = sum([cp.multiply(M[:,None,p], np.sum(ram_keep_tensors[p], axis=1, keepdims=True)) for p in range(Y)])
            lm2 = sum([cp.sum(cp.multiply(SM[p][:,:], -ram_keep_tensors[p]), axis=1, keepdims=True) for p in range(Y)])

            m1 = gradm + sm1 + lm1 + lm2
            # Forward checkpoint constraint
            constraints.append(m1 <= budget)

            m2 = gradm + wm + rm + sm2 + lm1 + lm2
            # Forward recomputation constraint
            constraints.append(m2 <= budget)

            for t in range(T):
                bkwd_t = V + t - 1
                bwd_node = self.si.nodes[bkwd_t]
                for p in range(len(bwd_node.args)):
                    if j in bwd_node.args[p]:
                        ram_keep_tensors[p][t,j] = ram_keep_tensors[p][t,j] + self.ram[j]

        for t in range(T):
            bkwd_t = V + t - 1
            bwd_node = self.si.nodes[bkwd_t]
            bwd_local_memory = self.local_memory(bkwd_t) if t!=0 else 0
            bwd_gradm = bwd_local_memory + self.fixed_ram(bkwd_t) + self.ram[bkwd_t]
            # for p in range(len(bwd_node.args)):
            if t>0:
                if len(RF) and bkwd_t in self.si.conv_list:
                    wmem = self.cmem(bkwd_t)
                    bwd_wm = cp.sum([RF[c][t,self.si.conv_list[bkwd_t]]*wmem[c] for c in range(len(wmem))])
                else:
                    wmem = self.cmem(bkwd_t)
                    bwd_wm = cp.sum([M[t,p]*wmem[p] for p in range(len(bwd_node.args))])
                lm3 = sum([M[t,p]*np.sum(ram_keep_tensors[p], axis=1)[t] for p in range(len(bwd_node.args))])
                sm3_ram = [self.ram.copy() for p in range(len(bwd_node.args))]
                for p in range(len(bwd_node.args)):
                    if len(bwd_node.dep_list_fwd[p]):
                        sm3_ram[p][bwd_node.dep_list_fwd[p]] = 0
                lm4 = sum([cp.sum(cp.multiply(SM[p][t,:],sm3_ram[p][:V])) for p in range(len(bwd_node.args))])
                m3 = bwd_wm + bwd_gradm + lm3 + lm4
                # Backward constraint
                constraints.append(m3 <= budget)

        # Objective
        if len(RF):
            fwd_compute = cp.sum([R[0, i] * self.cpu[i][0] for i in range(V) if i not in self.si.conv_list])
            fwd_recompute = cp.sum(sum([cp.multiply(R[1:, i],self.cpu_recompute[i][0]) for i in range(V) if i not in self.si.conv_list]))
            conv_fwd_compute = cp.sum(sum([cp.multiply(RF[c][:,self.si.conv_list[i]], self.cpu[i][c]) for i in self.si.conv_list if i<=self.si.loss for c in range(len(self.cpu[i]))])) # conv's compute and recompute same
            if self.si.do_inplace:
                inplace_fwd_compute = cp.sum(sum([cp.multiply(IP[:,i],self.cpu_inplace[i][0]-self.cpu[i][0]) for i in self.si.inplace_list])) # relu's compute and recompute same
            conv_bwd_compute = cp.sum(sum([cp.multiply(RF[c][:,self.si.conv_list[i]], self.cpu[i][c]) for i in self.si.conv_list if i>self.si.loss for c in range(len(self.cpu[i]))])) # conv's compute and recompute same
            bwd_compute = cp.sum([M[t, p] * self.cpu[t+V-1][p] for t in range(T) for p in range(Y) if (p<len(self.cpu[t+V-1]) and (t+V-1) not in self.si.conv_list)])
            if self.si.do_inplace:
                compute_fwd = fwd_compute + fwd_recompute + conv_fwd_compute + inplace_fwd_compute
            else:
                compute_fwd = fwd_compute + fwd_recompute + conv_fwd_compute
            compute_bwd = conv_bwd_compute + bwd_compute
            compute_cost = compute_fwd + compute_bwd
        else:
            fwd_compute = cp.sum([R[0, i] * self.cpu[i][0] for i in range(V)])
            fwd_recompute = cp.sum([R[t, i] * self.cpu_recompute[i][0] for t in range(1,T) for i in range(V)])
            if self.si.do_inplace:
                inplace_fwd_compute = cp.sum(sum([cp.multiply(IP[:,i],self.cpu_inplace[i][0]-self.cpu[i][0]) for i in self.si.inplace_list])) # relu's compute and recompute same
            bwd_compute = cp.sum([M[t, p] * self.cpu[t+V-1][p] for t in range(T) for p in range(Y) if p<len(self.cpu[t+V-1])])
            if self.si.do_inplace:
                compute_fwd = fwd_compute + fwd_recompute + inplace_fwd_compute
            else:
                compute_fwd = fwd_compute + fwd_recompute
            compute_bwd = bwd_compute
            compute_cost = compute_fwd + compute_bwd

        objective = cp.Minimize(compute_cost)

        # Solve
        problem = cp.Problem(objective, constraints)
        installed_solvers = cp.installed_solvers()
        from time import time
        solve_time = -1
        if "GUROBI" in self.solver:
            t0 = time()
            problem.solve(verbose=True, solver=cp.GUROBI, Threads=12, TimeLimit=self.time_limit, Method=1)
            solve_time = time() - t0
        if "CBC" in self.solver:
            t0 = time()
            problem.solve(verbose=True, solver=cp.CBC, numberThreads=12, maximumSeconds=self.time_limit)
            solve_time = time() - t0

        RFsol = np.zeros((self.si.num_conv_algos, T, len(self.si.conv_list)))
        if len(RF):
            for c in range(self.si.num_conv_algos):
                RFsol[c] = RF[c].value
        RFsol = RFsol.transpose((1,2,0))
        IPsol = np.zeros((T,V))
        if self.si.do_inplace:
            IPsol = IP.value
        solution = Solution(np.around(R.value), np.around(S.value), np.around(M.value), np.around(RFsol), np.around(IPsol), np.around(P.value), solve_time, compute_fwd.value, compute_bwd.value, compute_cost.value)

        fname = '../data/solution_%s_%4.2f' % (self.si.data_path, self.budget)
        if self.ablation:
            fname = fname+"_ablation"
        fname = fname+".pkl"
        if problem.status in ["infeasible", "unbounded"]:
            raise ValueError("Model infeasible")
        else:
           import pickle
           solfile = open(fname, 'wb')
           pickle.dump([self.si, solution], solfile)
           solfile.close()
        return solution

if __name__ == '__main__':
    import argparse
    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('bs')
    parser.add_argument('budget')
    parser.add_argument('mode')
    parser.add_argument('solver')
    parser.add_argument(
    "--ablation", action="store_true",
    help="Only solve with intermediate deps.")
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
    solver_model = Model(solver_info, budget, time_limit, args.solver, args.ablation)
    solution = solver_model.solve()
