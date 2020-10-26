import torch
from torch.utils.cpp_extension import load
from pathlib import Path

this_dir = Path(__file__).parent
pack_cpp = load(name="pack_cpp", sources=[this_dir / "pack.cpp", this_dir / "pack.cu"], extra_cflags=['-std=c++17'])


def pack(v: torch.Tensor) -> torch.Tensor:
    """
    Pack 8 boolean values into a uint8.

    :param v: A 1D bool tensor
    :return: A 1D uint8 tensor of 8x smaller size
    """
    return pack_cpp.pack(v)

def unpack(p: torch.Tensor, n: int = 0) -> torch.Tensor:
    """
    Unpack a uint8 into 8 boolean values.

    :param p: A 1D uint8 tensor
    :param n: The output size (default v.size(0)*8)
    :return: A 1D bool tensor of size N
    """
    return pack_cpp.unpack(p, n)


def unpack_multiply(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Unpack a uint8 into 8 boolean values and multiply them with another tensor.

    :param p: A 1D uint8 tensor
    :param v: A 1D tensor
    :return: A 1D tensor of the same type and shape as v
    """
    return pack_cpp.unpack_multiply(p, v)

def pack_two(v: torch.Tensor) -> torch.Tensor:
    """
    Pack 2 uint8 values (<16) into a single uint8.
    :param v: A 1D uint8 tensor
    :return: A 1D uint8 tensor of 2x smaller size
    """
    return pack_cpp.pack_two(v)

def unpack_two(v: torch.Tensor, n: int = 0) -> torch.Tensor:
    """
    Unpack a uint8 into 2 uint8 values <16.
    :param v: A 1D uint8 tensor
    :param n: The output size (default v.size(0)*2)
    :return: A 1D uint8 tensor of size N*2
    """
    return pack_cpp.unpack_two(v, n)


if __name__ == "__main__":
    rnd = torch.rand(10)
    t = rnd > 0.5
    p = pack(t)
    print(t)
    print(p)
    print(unpack(p, t.size(0)))
    print(unpack_multiply(p, rnd))

    print('*'*10, 'benchmarking', '*'*10)
    rnd = torch.rand(128*64*64*64)
    v = rnd > 0.5
    from time import time
    for d in [torch.device('cuda')]: # torch.device('cpu'),
        rnd = rnd.to(d)
        v = v.to(d)
        print(d)
        t0 = time()
        for i in range(1000):
            p = pack(v)
        print('pack', time()-t0)
        t0 = time()
        for i in range(1000):
            vv = unpack(p, v.size(0))
        print('unpack', time()-t0)
        print('correct', bool((v == vv).all()))

        # Just in case pytorch or cuda optimize a bit too aggressively
        t0 = time()
        for i in range(1000):
            p = pack(v)
            v = unpack(p, v.size(0))
        print('pack and unpack', time()-t0)

        t0 = time()
        r2 = rnd
        for i in range(1000):
            r2 = unpack_multiply(p, r2)
        print('unpack multiply', time()-t0)

        # Compare to thresholding and relu
        t0 = time()
        for i in range(1000):
            rnd = torch.relu(rnd)
        print('relu', time()-t0)

        # Compare to thresholding and relu
        t0 = time()
        for i in range(1000):
            v = rnd > 0.5
        print('ge', time()-t0)