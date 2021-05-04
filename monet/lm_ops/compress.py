import torch
from torch.utils.cpp_extension import load
from pathlib import Path

this_dir = Path(__file__).parent
compress_cpp = load(name="compress_cpp", sources=[this_dir / "compress.cpp", this_dir / "compress.cu"], extra_cflags=['-std=c++17', '-lcusparse'], extra_cuda_cflags=['-lcusparse'],extra_ldflags=['-lcusparse'])

def compress_csr_256(ip: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Return a list of dense vectors and their indices 

    :param ip: 1D float32 tensor - sparse form activation
    :return0: 1D float32 tensor - dense form CSR activation
    :return1: 1D int32 tensor - CSR indices

    """
    d1,d2,d3,d4=ip.shape
    ip_dim = d1*d2*d3*d4
    pad_val = (256 - ip_dim%256)%256
    if pad_val != 0:
        ip_new = torch.nn.functional.pad(ip.view(ip_dim), (0,pad_val))
    else:
        ip_new = ip
    del ip
    ip_new = ip_new.view(-1,256)
    nzrow = (ip_new!=0).sum(dim=1)
    nz = nzrow.sum().item()
    nzrow = nzrow.to(torch.int32)
#    print(ip_new)
#    print(ip_new.shape, nzrow, nz)
    return compress_cpp.compress_csr_256(ip_new, nzrow, nz)

def uncompress_csr_256(compip: torch.Tensor, indx: torch.Tensor, row: torch.Tensor, N: int) -> torch.Tensor:
    """
    Returns an uncompressed dense vector

    :param compip: 1D float32 tensor - CSR compressed input
    :param indx: 1D int32 tensor - CSR indices
    :return: 1D float32 tensor - uncompressed activation
    """
    return compress_cpp.uncompress_csr_256(compip, indx, row, N)
