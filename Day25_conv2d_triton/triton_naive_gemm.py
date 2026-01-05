import os
import torch
import triton
import triton.language as tl
from src.utils import ceil_div

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2026)
#os.environ['TRITON_INTERPRET']='0'

@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    '''Naive GeMM kernel'''
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_1d_m = pid0*bm + tl.arange(0, bm)
    offset_1d_n = pid1*bn + tl.arange(0, bn)

    offset_2d_c = tl.expand_dims(offset_1d_m,1)*N + tl.expand_dims(offset_1d_n,0)  

    mask_1d_m = offset_1d_m < M
    mask_1d_n = offset_1d_n < N
    mask_2d_c = tl.expand_dims(mask_1d_m, 1) & tl.expand_dims(mask_1d_n, 0)

    p_val = tl.zeros((bm, bn), dtype=tl.float32)

    num_blocks = (K + bk - 1) // bk
    for block_idx in range(num_blocks):
        k_start = block_idx * bk
        offset_1d_k = k_start + tl.arange(0, bk)

        offset_2d_a = tl.expand_dims(offset_1d_m,1)*K + tl.expand_dims(offset_1d_k,0)
        offset_2d_b = tl.expand_dims(offset_1d_k,1)*N + tl.expand_dims(offset_1d_n,0)

        mask_1d_k = offset_1d_k < K
        mask_a = tl.expand_dims(mask_1d_m,1) & tl.expand_dims(mask_1d_k,0)
        mask_b = tl.expand_dims(mask_1d_k,1) & tl.expand_dims(mask_1d_n,0)

        a_val = tl.load(a_ptr + offset_2d_a, mask=mask_a, other=0.0)
        b_val = tl.load(b_ptr + offset_2d_b, mask=mask_b, other=0.0)

        p_val += tl.dot(a_val, b_val)

    tl.store(c_ptr + offset_2d_c, p_val, mask=mask_2d_c)



def gemm_custom(a: torch.Tensor, b: torch.Tensor, M:int, N:int, K:int) -> torch.Tensor:
    bm = 16
    bn = 16
    bk = 16

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (ceil_div(M,bm), ceil_div(N,bn))

    gemm_kernel[grid](a, b, c, M, N, K, bm, bn, bk)

    return c

def main():
    M = 512
    K = 512
    N = 512

    a = torch.randn((M,K), dtype = torch.float32, device=device)
    b = torch.randn((K,N), dtype = torch.float32, device=device)
    c_pytorch = a @ b

    c_triton = gemm_custom(a, b, M, N, K)

    goldens_match = torch.allclose(c_pytorch, c_triton, rtol=1e-2, atol=1e-2)
    if(goldens_match):
        print(f'Success')
    else:
        mismatch_locations = torch.argwhere(c_pytorch != c_triton).tolist()
        for idx in mismatch_locations:
            print(f'Mismatch at {idx[0]}, {idx[1]} \
                , {c_pytorch[idx[0],idx[1]]} != {c_triton[idx[0],idx[1]]}')

if __name__ == '__main__':
    main()