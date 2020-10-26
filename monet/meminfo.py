import torch
import monet.lm_ops as lm_ops
from monet.graph import *

# Get compute information for a node
def computeinfo(n, op, graph, bs, bwd_op, conv_mode, inplace_mode, *args):
    from time import time
    import numpy as np
    params = args
    runtime_sample=20

    if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm":
        if bwd_op == "param_grad":
            args = [a.value if isinstance(a, ComputeNode.V) else
                    params[a.index].requires_grad_(True) if isinstance(graph.nodes[a.index], Param) else
                    torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(False).cuda() for a in n.args]
            args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, True, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], False, torch.cuda.current_device()] for a in n.args]
        else:
            args = [a.value if isinstance(a, ComputeNode.V) else
                    params[a.index].requires_grad_(False) if isinstance(graph.nodes[a.index], Param) else
                    torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(a.requires_grad).cuda() for a in n.args]
            args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, False, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], a.requires_grad, torch.cuda.current_device()] for a in n.args]
    else:
        args = [a.value if isinstance(a, ComputeNode.V) else
                params[a.index] if isinstance(graph.nodes[a.index], Param) else
                torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(a.requires_grad).cuda() for a in n.args]

        args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, params[a.index].requires_grad, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], a.requires_grad, torch.cuda.current_device()] for a in n.args]
    runtime_fwd_recompute = []
    runtime_fwd = []
    # Forward runtime
    if conv_mode and (n.op == "aten::_convolution" and not n.is_depthwise):
        for i in range(op.n_fwd_algos()):
            op.algorithm = i
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t0 = time()
                for rs in range(runtime_sample):
                    outs = op.forward(*args)
                    del outs
                torch.cuda.synchronize()
                runtime = (time() - t0)/runtime_sample
                runtime_fwd.append(runtime)
                torch.cuda.empty_cache()
            except Exception as e:
                print("C: Failed fwd algo", i, e)
                runtime_fwd.append(-1)
                torch.cuda.empty_cache()
        outs = torch.zeros([val if val!=-1 else bs for val in list(n.shape)]).to(args[0].device)
    elif inplace_mode and hasattr(op, 'inplace'):
        outs = torch.zeros(1)
        for do_inplace in [False, True]:
            op.inplace = do_inplace
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time()
            for rs in range(runtime_sample):
                del outs
                outs = op.forward(*args)
            torch.cuda.synchronize()
            runtime = (time() - t0)/runtime_sample
            runtime_fwd.append(runtime)
            torch.cuda.empty_cache()
    elif n.op == "aten::batch_norm":
        outs = torch.zeros(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time()
        for rs in range(runtime_sample):
            del outs
            outs = op.forward(*args)    # Test BN fwd as if it is happening for first time
            op.params = None
        torch.cuda.synchronize()
        runtime_fwd.append((time()-t0) / runtime_sample)
        outs = op.forward(*args)    # Seed op with BN statistics
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time()
        for rs in range(runtime_sample):
            outs = op.forward(*args)    # Test BN fwd with already computed statistics
        torch.cuda.synchronize()
        runtime_fwd_recompute.append((time()-t0) / runtime_sample)
    else:
        if n.is_depthwise:
            op.is_depthwise = True
        outs = torch.zeros(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time()
        for rs in range(runtime_sample):
            del outs
            outs = op.forward(*args)
        torch.cuda.synchronize()
        runtime_fwd.append((time()-t0) / runtime_sample)

    # Backward runtime
    runtime_bwd = []
    if bwd_op is not None:
        if not isinstance(outs, (tuple, list)):
            outs = [outs]
        required_storage = []
        storage_list = op.backward_storage
        if not isinstance(storage_list, list):
            storage_list = [storage_list]
        for storage in storage_list:
            if isinstance(storage, lm_ops.InputStorage):
                for i in storage.ids:
                    required_storage.append(args_meta[i])
            elif isinstance(storage, lm_ops.OutputStorage):
                required_storage.append([outs[0].shape, outs[0].requires_grad, outs[0].device, outs[0].dtype])
            elif isinstance(storage, lm_ops.IntermediateStorage):
                assert len(outs) > 1
                required_storage.append([outs[-1].shape, outs[-1].requires_grad,outs[-1].device, outs[-1].dtype])

        device = outs[0].device
        shape = outs[0].shape
        for out in outs:
            del out
        del outs
        for arg in args:
            del arg
        del args
        if hasattr(op, 'params'):
            p = op.params

        stored = [i if not isinstance(i,list) else torch.ones(i[0], device=i[2], requires_grad=i[1], dtype=i[3] if len(i)>3 else torch.float) for i in required_storage]
        gradin = torch.ones(shape,device=device)
        if hasattr(op, 'params'):
            op.params = p

        if bwd_op == "ip_grad":
            if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm":
                stored[0] = None

        if conv_mode and (n.op == "aten::_convolution" and not n.is_depthwise):
            if bwd_op == "ip_grad":
                step = 10
                bwdtypes = op.n_bwd_ip_algos()
            else:
                step = 0
                bwdtypes = op.n_bwd_wt_algos()
            for bwdtype in range(bwdtypes):
                try:
                    op.algorithm = bwdtype+step
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    t0 = time()
                    for rs in range(runtime_sample):
                        grads = op.backward(gradin, stored, nodel=True)
                        del grads
                    torch.cuda.synchronize(device)
                    runtime = (time() - t0)/runtime_sample
                    runtime_bwd.append(runtime)
                except Exception as e:
                    print("C: Failed bwd algo", bwdtype,e)
                    runtime_bwd.append(-1)
                    torch.cuda.empty_cache()
        else:
            if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm":
                if bwd_op == "param_grad":
                    convtype = 0
                else:
                    convtype = 10
                op.algorithm = convtype
            if n.is_depthwise:
                op.is_depthwise = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time()
            for n in range(runtime_sample):
                grads = op.backward(gradin, stored, nodel=True)
                del grads
            torch.cuda.synchronize()
            runtime = (time()-t0) / runtime_sample
            runtime_bwd.append(runtime)

    runtime_fwd = np.array(runtime_fwd, dtype='f4') * 1000.
    runtime_fwd_recompute = np.array(runtime_fwd_recompute, dtype='f4') * 1000.
    runtime_bwd = np.array(runtime_bwd, dtype='f4') * 1000.
    return runtime_fwd, runtime_bwd, runtime_fwd_recompute


# Get workspace memory information for a node
def meminfo(n, op, graph, bs, bwd_op, conv_mode, inplace_mode, *args1):
    from time import time
    import numpy as np
    params = args1

    if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm":
        if bwd_op == "param_grad":
            args = [a.value if isinstance(a, ComputeNode.V) else
                    params[a.index].requires_grad_(True) if isinstance(graph.nodes[a.index], Param) else
                    torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(False).cuda() for a in n.args]
            args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, True, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], False, torch.cuda.current_device()] for a in n.args]
        else:
            args = [a.value if isinstance(a, ComputeNode.V) else
                    params[a.index].requires_grad_(False) if isinstance(graph.nodes[a.index], Param) else
                    torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(a.requires_grad).cuda() for a in n.args]
            args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, False, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], a.requires_grad, torch.cuda.current_device()] for a in n.args]
    else:
        args = [a.value if isinstance(a, ComputeNode.V) else
                params[a.index] if isinstance(graph.nodes[a.index], Param) else
                torch.randn([val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)]).requires_grad_(a.requires_grad).cuda() for a in n.args]

        args_meta = [ a.value if isinstance(a, ComputeNode.V) else
                    [params[a.index].shape, params[a.index].requires_grad, params[a.index].device] if isinstance(graph.nodes[a.index], Param) else
                    [ [val if val!=-1 else bs for val in list(graph.nodes[a.index].shape)], a.requires_grad, torch.cuda.current_device()] for a in n.args]

    # Recompute memory
    fwd_working_memory_recompute = []

    fwd_working_memory = []

    # Forward runtime
    if conv_mode and (n.op == "aten::_convolution" and not n.is_depthwise):
        for i in range(op.n_fwd_algos()):
            op.algorithm = i
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_stats()
                outs = op.forward(*args)
                torch.cuda.empty_cache()
                m1 = torch.cuda.memory_stats()
                peak_recompute = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
                if not isinstance(outs, (tuple, list)):
                    outs = [outs]
                out_mem = 0
                for out in outs:
                    if out is not None:
                        out_mem = out_mem + np.prod(list(out.shape))*out.element_size()
                fwd_working_memory.append(peak_recompute - out_mem)
                del outs, out
            except Exception as e:
                print("M: Failed fwd algo", i, e)
                fwd_working_memory.append(0)
                torch.cuda.empty_cache()
        outs = [torch.zeros([val if val!=-1 else bs for val in list(n.shape)]).to(args[0].device)]
    elif inplace_mode and hasattr(op, 'inplace'):
        outs = torch.zeros(1)
        for do_inplace in [False, True]:
            op.inplace = do_inplace
            del outs
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_stats()
            outs = op.forward(*args)
            torch.cuda.empty_cache()
            m1 = torch.cuda.memory_stats()
            peak_recompute = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
            if not isinstance(outs, (tuple, list)):
                outs = [outs]
            out_mem = 0
            for out in outs:
                if out is not None:
                    out_mem = out_mem + np.prod(list(out.shape))*out.element_size()
            fwd_working_memory.append(peak_recompute - out_mem)
            del out
    elif n.op == "aten::batch_norm":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_stats()
        outs = op.forward(*args)    # Test BN fwd as if it is happening for first time
        torch.cuda.empty_cache()
        m1 = torch.cuda.memory_stats()
        peak_recompute = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
        if not isinstance(outs, (tuple, list)):
            outs = [outs]
        out_mem = 0
        for out in outs:
            if out is not None:
                out_mem = out_mem + np.prod(list(out.shape))*out.element_size()
        fwd_working_memory.append(peak_recompute - out_mem)
        del out

        del outs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_stats()
        outs = op.forward(*args)    # Repeat BN using same op
        torch.cuda.empty_cache()
        m1 = torch.cuda.memory_stats()
        peak_recompute = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
        if not isinstance(outs, (tuple, list)):
            outs = [outs]
        out_mem = 0
        for out in outs:
            if out is not None:
                out_mem = out_mem + np.prod(list(out.shape))*out.element_size()
        fwd_working_memory_recompute.append(peak_recompute - out_mem)
        del out
    else:
        if n.is_depthwise:
            op.is_depthwise = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_stats()
        outs = op.forward(*args)
        torch.cuda.empty_cache()
        m1 = torch.cuda.memory_stats()
        peak_recompute = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
        if not isinstance(outs, (tuple, list)):
            outs = [outs]
        out_mem = 0
        for out in outs:
            if out is not None:
                out_mem = out_mem + np.prod(list(out.shape))*out.element_size()
        fwd_working_memory.append(peak_recompute - out_mem)
        del out


    bwd_working_memory = []
    if bwd_op is not None:
        required_storage = []
        storage_list = op.backward_storage
        if not isinstance(storage_list, list):
            storage_list = [storage_list]
        for storage in storage_list:
            if isinstance(storage, lm_ops.InputStorage):
                for i in storage.ids:
                    required_storage.append(args_meta[i])
            elif isinstance(storage, lm_ops.OutputStorage):
                required_storage.append([outs[0].shape, outs[0].requires_grad, outs[0].device, outs[0].dtype])
            elif isinstance(storage, lm_ops.IntermediateStorage):
                assert len(outs) > 1
                required_storage.append([outs[-1].shape, outs[-1].requires_grad,outs[-1].device, outs[-1].dtype])
        device = outs[0].device
        shape = outs[0].shape
        for out in outs:
            del out
        del outs
        for arg in args:
            del arg
        del args

        # Backward peak
        typebwds = 1
        convtype = -1
        if n.op == "aten::addmm" or (n.op == "aten::_convolution" and not n.is_depthwise):
            if bwd_op == "ip_grad":
                convtype = 10
            else:
                convtype = 0
        if conv_mode and (n.op == "aten::_convolution" and not n.is_depthwise):
            if bwd_op == "ip_grad":
                typebwds = op.n_bwd_ip_algos()
            else:
                typebwds = op.n_bwd_wt_algos()
        if bwd_op == "ip_grad":
            if (n.op == "aten::_convolution" and not n.is_depthwise) or n.op == "aten::addmm":
                required_storage[0] = None

        for typebwd in range(typebwds):
            try:
                stored = [i if not isinstance(i,list) else torch.ones(i[0], device=i[2], requires_grad=i[1], dtype=i[3] if len(i)>3 else torch.float) for i in required_storage]
                gradin = torch.ones(shape,device=device)
                if convtype != -1:
                    op.algorithm = convtype + typebwd
                if n.is_depthwise:
                    op.is_depthwise = True
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_stats()
                # required_storage will get deleted inside and we want to include this in our measurement
                grads = op.backward(gradin, stored)
                m1 = torch.cuda.memory_stats()
                peak_bwd = m1['allocated_bytes.all.peak']-m0['allocated_bytes.all.peak']
                torch.cuda.empty_cache()
                out_mem = 0
                for gradi in grads:
                    if gradi is not None:
                        out_mem = out_mem + np.prod(list(gradi.shape))*gradi.element_size()
                bwd_working_memory.append(peak_bwd - out_mem)
                del gradin, grads
            except Exception as e:
                print("M: Failed bwd algo", typebwd, e)
                if typebwds == 1:
                    raise RuntimeError("Should have passed for ops with single impl")
                bwd_working_memory.append(0)

    fwd_working_memory = np.array(fwd_working_memory) / 1024. / 1024.
    fwd_working_memory_recompute = np.array(fwd_working_memory_recompute) / 1024. / 1024.
    bwd_working_memory = np.array(bwd_working_memory) / 1024. / 1024.

    return fwd_working_memory, bwd_working_memory, fwd_working_memory_recompute
