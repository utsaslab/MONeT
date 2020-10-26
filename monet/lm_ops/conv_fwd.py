import torch
from torch.utils.cpp_extension import load
from pathlib import Path
from time import time

this_dir = Path(__file__).parent
conv_fwd_cpp = load(name="conv_fwd_cpp", sources=[this_dir/"conv.cpp"], extra_cflags=['-std=c++17'], extra_include_paths=[str(this_dir)], with_cuda=True)
cudnn_convolution = conv_fwd_cpp.cudnn_convolution
cudnn_convolution_backward_input = conv_fwd_cpp.cudnn_convolution_backward_input
cudnn_convolution_backward_weight = conv_fwd_cpp.cudnn_convolution_backward_weight


if __name__ == '__main__':
    i = torch.zeros((256, 64, 56, 56), device='cuda')
    w = torch.zeros((64, 64, 3, 3), device='cuda')
    print('fwd')
    for t in range(conv_fwd_cpp.n_fwd_algos()):
        torch.cuda.reset_peak_memory_stats()
        t0 = time()
        try:
            for it in range(10):
                o = cudnn_convolution(i, w, (1, 1), (1, 1), (1, 1), 1, t)
        except Exception as e:
            print('%02d   failed' % t, e)
        else:
            torch.cuda.synchronize()
            M = torch.cuda.memory_stats()
            print('%02d   %6.3f s   %0.3f GB' % (t, time() - t0, M["allocated_bytes.all.peak"] / 1024. / 1024. / 1024.))

    print('bwd')
    for t in range(conv_fwd_cpp.n_bwd_ip_algos()):
        torch.cuda.reset_peak_memory_stats()
        t0 = time()
        try:
            for it in range(10):
                cudnn_convolution_backward_input(i.shape, o, w, (1, 1), (1, 1), (1, 1), 1, t)
        except Exception as e:
            print('%02d   failed' % t, e)
        else:
            torch.cuda.synchronize()
            M = torch.cuda.memory_stats()
            print('%02d   %6.3f s   %0.3f GB' % (t, time() - t0, M["allocated_bytes.all.peak"] / 1024. / 1024. / 1024.))

    print('bwd weight')
    for t in range(conv_fwd_cpp.n_bwd_wt_algos()):
        torch.cuda.reset_peak_memory_stats()
        t0 = time()
        try:
            for it in range(10):
                cudnn_convolution_backward_weight(w.shape, o, i, (1, 1), (1, 1), (1, 1), 1, t)
        except Exception as e:
            print('%02d   failed' % t, e)
        else:
            torch.cuda.synchronize()
            M = torch.cuda.memory_stats()
            print('%02d   %6.3f s   %0.3f GB' % (t, time() - t0, M["allocated_bytes.all.peak"] / 1024. / 1024. / 1024.))
