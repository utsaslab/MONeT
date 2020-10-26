from typing import Set

import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.utils.definitions import Vertex

SOLVER_DTYPE = np.int


def setup_implied_s_backwards(g: DFGraph, s: np.ndarray = None):
    """
    Given a backward graph, this function will set the appropriate items in S to 1 in order
    to satisfy no-recompute rules during backwards optimization.
    """
    s = s if s is not None else np.zeros((g.size, g.size), dtype=SOLVER_DTYPE)
    for (start, end) in g.induce_subgraph(g.vbwd):
        for t in range(start + 1, end + 1):
            s[t, start] = 1
    return s


def gen_s_matrix_fixed_checkpoints(g: DFGraph, segment_set: Set[Vertex]):
    """
    Given a list of checkpoint locations, this function will generate
    as output S matrices denoting checkpoint schedule, given a set of
    fixed segments (only recompute once).
    """
    T = len(g.vfwd)
    Ttotal = g.size
    segment_set = list(sorted(segment_set))
    S = np.zeros((g.size, g.size), dtype=SOLVER_DTYPE)
    # set minimum input requirements
    for v in g.v:
        for u in g.predecessors(v):
            for t in range(u + 1, v):
                S[t, u] = 1

    # stripe every k nodes
    for t in range(1, Ttotal):
        for i in segment_set:
            if i < t:
                S[t, i] = 1

    # checkpoint ladders
    starts = [0] + list(map(lambda x: x, segment_set))
    ends = segment_set + [T + 1]
    for start, end in zip(starts, ends):
        for t in filter(lambda t: t < Ttotal, map(lambda x: Ttotal - x - 1, range(start, end))):
            for i in range(start, min(t, end)):
                S[t, i] = 1

    # forward checkpoint block
    for start, end in zip(starts, ends):
        for t in filter(lambda t: t < Ttotal, range(start, end + 1)):
            for i in range(start, min(t, end)):
                S[t, i] = 1

    # backward checkpoint block
    # originally used as baselines will checkpoint whole blocks (e.g. Chen 2016 checkpoints entire backwards blocks),
    # but removed in public release as schedules are faster without this.
    # for start, end in zip(starts, ends):
    #     for t in filter(lambda _t: _t < Ttotal, range(start, end + 1)):
    #         back_t = Ttotal - 1 - t
    #         for i in range(start, end):
    #             back_i = g.forward_to_backward(i)
    #             if back_i is not None and back_i < back_t:
    #                 S[back_t, back_i] = 1

    S = setup_implied_s_backwards(g, S)
    return S


def solve_r_opt(g: DFGraph, s: np.ndarray):
    """Find the optimal recomputation pattern given caching decisions.
    Given S, E = [(i, j)] where node j depends on the result of node i,
    find R that minimizes cost, satisfies constraints. Assumes recomputation
    costs are nonnegative.
    NOTE: Does NOT check if memory limits are exceeded.
    Enforcing R[t,i] != S[t,i] does not seem to be necessary.
    """
    T = s.shape[0]
    assert s.shape[1] == T

    R = np.eye(T, dtype=s.dtype)  # Enforce R_t,t = 1
    # Enforce S_{t+1,v} <= S_{t,v} + R_{t,v},
    # i.e. R_{t,v} >= S_{t+1,v} - S_{t,v}
    sdiff = s[1:] - s[:-1]
    R[:-1] = R[:-1] | (R[:-1] < sdiff)
    # Create reverse adjacency list (child -> parents, i.e. node -> dependencies)
    adj = [[] for _ in range(T)]
    for (u, v) in g.edge_list:
        adj[v].append(u)
    # Enforce R_{t,v} <= R_{t,u} + S_{t,u} for all (u, v) \in E
    for t in range(T):
        for v in range(t, -1, -1):
            for u in adj[v]:
                if R[t, v] > R[t, u] + s[t, u]:
                    R[t, u] = 1
    return R