import torch
import triton
import triton.language as tl
from src.utils import offset_1d, offset_2d


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, bm, bn, bk, M, N, K):
    '''Naive GeMM kernel'''
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_m = pid0*bm*K*tl.arange(bm)
    offset_n = pid1*bn*tl.arange(bn)
    offset_k = tl.arange(bk)

    a_index = a_ptr + 

    for idx_k in range(0, K, bk):
        a_block = tl.load()






def gemm_custom(a: torch.Tensor, b: torch.Tensor, M:int, N:int, K:int) -> torch.Tensor:
    bm = 2
    bn = 2
    bk = 2

    c = torch.empty(M, N)
    grid = lambda meta:((M/bm) , (N/bn))

    gemm_kernel[grid](a, b, c, bm, bn, bk, M, N, K)
    c.view(M, N)

    return c

def main():
    M = 4
    K = 4
    N = 4

    a = torch.arange(M*K).view(M, K)
    b = torch.arange(K*N).view(K, N)
    c_pytorch = a @ b

    c_triton = gemm_custom(a, b, M, N, K)

    goldens_match = torch.allclose(c_pytorch, c_triton, atol = 1e-4)
    if(goldens_match):
        print(f'Success')
    else:
        mismatch_locations = torch.argwhere(c_pytorch != c_triton).tolist()
        for idx in mismatch_locations:
            print(f'Mismatch at {idx[0]}, {idx[1]} \
                , {c_pytorch[idx[0],idx[1]]} != {c_triton[idx[0],idx[1]]}')

if __name__ == '__main__':
    main()