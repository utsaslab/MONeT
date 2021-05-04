import torch
from functools import lru_cache


class Node:
    def __init__(self, shape):
        self.shape = shape


class Input(Node):
    def __repr__(self):
        return '<Input %s>' % str(list(self.shape))


class Param(Node):
    def __repr__(self):
        return '<Param %s>' % str(list(self.shape))


class ComputeNode(Node):
    class Arg:
        pass

    class V(Arg):
        def __init__(self, v):
            self.value = v

        def __repr__(self):
            return '<V %s>' % str(self.value)

    class D(Arg):
        def __init__(self, index, requires_grad=False):
            self.index = index
            self.requires_grad = requires_grad

        def __repr__(self):
            return '<D %d %d>' % (self.index, self.requires_grad)

    def __init__(self, shape, nodeid, op, args, has_backward, is_depthwise=False, compress_conv=-1):
        super().__init__(shape)
        self._op = op
        self._args = args
        self.id = nodeid
        self._has_backward = has_backward
        self._is_depthwise = is_depthwise
        self.compress_conv = compress_conv

    @property
    @lru_cache(maxsize=512)
    def op(self):
        return self._op

    def clear(self):
        ComputeNode.op.fget.cache_clear()

    @property
    def args(self):
        return self._args

    @property
    def has_backward(self):
        return self._has_backward

    @property
    @lru_cache(maxsize=512)
    def is_depthwise(self):
        return self._is_depthwise

    @property
    @lru_cache(maxsize=128)
    def dependencies(self):
        return [(a.index, a.requires_grad) for a in self._args if isinstance(a, self.D)]

    def __repr__(self):
        return '<Op %s %s>' % (str(self._op), str(list(self.shape)))


class Graph:
    def __init__(self):
        self._nodes = []
        self._outputs = []

    def _add_node(self, node):
        self._nodes.append(node)
        return len(self._nodes)-1

    def _add_input(self, shape):
        return self._add_node(Input(shape))

    def _add_param(self, shape):
        return self._add_node(Param(shape))

    def _add_op(self, shape, op, args, has_backward=False, is_depthwise=False, compress_conv=-1):
        nodeid = len(self._nodes)
        return self._add_node(ComputeNode(shape, nodeid, op, args, has_backward, is_depthwise, compress_conv))

    def _add_output(self, output_id):
        self._outputs.append(output_id)

    @property
    def nodes(self):
        return self._nodes

    @classmethod
    def create(cls, model, input_shape=(3, 224, 224)):
        # create a graph of the forward pass
        # JIT trace the model
        args = (torch.ones((23,) + input_shape),)
        graph, torch_out = torch.jit._get_trace_graph(model, args, _force_outplace=False, _return_inputs_states=False)
        torch._C._jit_pass_constant_propagation(graph)
        torch._C._jit_pass_inline(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        params = torch.jit._unique_state_dict(model)

        assert len(list(graph.inputs())) == len(args) + len(params)
        node_id = {}
        r = cls()
        arg_and_param_shape = [list(a.shape) for a in args] + [list(p.shape) for p in params.values()]
        for k, i in enumerate(graph.inputs()):
            if k < len(args):
                node_id[i.unique()] = r._add_input([-1]+arg_and_param_shape[k][1:])
            else:
                node_id[i.unique()] = r._add_param(arg_and_param_shape[k])

        const = {}

        # Track connected nodes in the graph
        track = set()
        track.add("input.1")
        for node in graph.nodes():
            if node.kind()!="aten::size":
                for ip in node.inputs():
                    if ip.debugName() in track or "input" in ip.debugName():
                        track.add(node.output().debugName())
                if "input" in node.output().debugName():
                    track.add(node.output().debugName())

        list_contents = {}
        for n in graph.nodes():
            compress_conv = -1  # Compress storage of conv input because it is a ReLU output
            is_gist_mp = -1     # Compress max_pool indices if its input is a ReLU output
            assert n.kind() != 'prim::GetAttr'
            if n.kind() == 'prim::Constant':
                const[n.output().unique()] = n['value'] if n.hasAttribute('value') else None
            elif len(n.kind()) > 6 and n.kind()[:6] == 'aten::':
                args = []
                for i in n.inputs():
                    iu = i.unique()
                    if iu in list_contents:
                        iu_list = list_contents[iu]
                    else:
                        iu_list = [iu]
                    for iu in iu_list:
                        if iu in const:
                            args.append(ComputeNode.V(const[iu]))
                        elif iu in node_id:
                            if i.debugName() not in track and (not isinstance(r._nodes[node_id[iu]], Input)) and (not isinstance(r._nodes[node_id[iu]], Param)): # Doing this for addmm and transpose
                                for ii in i.node().inputs():
                                    iiu = ii.unique()
                                    assert (isinstance(r._nodes[node_id[iiu]], Input) or isinstance(r._nodes[node_id[iiu]], Param)) == True
                                    args.append(ComputeNode.D(node_id[iiu], ii.requires_grad()))
                            else:
                                if n.kind() == "aten::_convolution" and isinstance(r._nodes[node_id[iu]], ComputeNode) and r._nodes[node_id[iu]].op == 'aten::relu_':
                                    compress_conv = node_id[iu]
                                    r._nodes[node_id[iu]]._op = 'aten::nosave_relu_'
                                    r._nodes[node_id[iu]].clear()
                                elif n.kind() == "aten::_convolution" and isinstance(r._nodes[node_id[iu]], ComputeNode) and r._nodes[node_id[iu]].op == 'aten::relu':
                                    compress_conv = node_id[iu]
                                    r._nodes[node_id[iu]]._op = 'aten::nosave_relu'
                                    r._nodes[node_id[iu]].clear()
                                elif n.kind() == "aten::max_pool2d" and isinstance(r._nodes[node_id[iu]], ComputeNode) and r._nodes[node_id[iu]].op == 'aten::relu_':
                                    r._nodes[node_id[iu]]._op = 'aten::savesign_relu_'
                                    is_gist_mp = 1
                                    r._nodes[node_id[iu]].clear()
                                elif n.kind() == "aten::max_pool2d" and isinstance(r._nodes[node_id[iu]], ComputeNode) and r._nodes[node_id[iu]].op == 'aten::relu':
                                    r._nodes[node_id[iu]]._op = 'aten::savesign_relu'
                                    is_gist_mp = 1
                                    r._nodes[node_id[iu]].clear()
                                args.append(ComputeNode.D(node_id[iu], i.requires_grad()))
                        else:
                            raise ValueError('Nodes %s disconnected' % repr(i))
                has_backward = False
                if n.output().debugName() in track:
                    has_backward = True
                # Identify depthwise conv
                is_depthwise = False
                if n.kind() == "aten::_convolution":
                    assert isinstance(args[8], ComputeNode.V)
                    if args[8].value > 1 and args[8].value == r.nodes[args[0].index].shape[1]:
                        is_depthwise = True
                # Add node to graph
                kind = n.kind()
                if compress_conv != -1 or is_gist_mp != -1:
                    kind = kind[:6] + 'gist_' + kind[6:]
                node_id[n.output().unique()] = r._add_op([s if s != 23 else -1 for s in n.output().type().sizes()],
                                                         kind, args, has_backward, is_depthwise, compress_conv)

            elif n.kind() in ['prim::ListConstruct', 'prim::TupleConstruct']:
                list_contents[n.output().unique()] = [i.unique() for i in n.inputs()]
            else:
                print('Unknown OP', n.kind(), n)
        # Identify outputs
        for op in graph.outputs():
            if op.node().kind()[:6]  == 'aten::':
                r._add_output(node_id[op.unique()])
            elif op.node().kind() == 'prim::TupleConstruct':
                for i in op.node().inputs():
                    r._add_output(node_id[i.unique()])
        return r
