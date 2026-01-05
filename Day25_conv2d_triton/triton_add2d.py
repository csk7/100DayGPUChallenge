import torch
import triton
import triton.language as tl
from src.utils import breakpoint_if, print_if, ceil_div

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, M, N, bM: tl.constexpr, bN: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_1d_y = pid0*bM + tl.arange(0,bM)
    offset_1d_x = pid1*bN + tl.arange(0,bN)
    offset = tl.expand_dims(offset_1d_y, 1)*N + tl.expand_dims(offset_1d_x, 0)*1

    mask_M = offset_1d_y < M
    mask_N = offset_1d_x < N

    mask = tl.expand_dims(mask_M, 1) & tl.expand_dims(mask_N, 0)
    
    a_val = tl.load(a_ptr + offset, mask)
    b_val = tl.load(b_ptr + offset, mask)
    c_val = a_val + b_val

    tl.store(c_ptr + offset, c_val, mask)


def triton_add(a, b, c, M, N):
    bM = 2
    bN = 4

    grid = (ceil_div(M,bM), ceil_div(N,bN))

    add_kernel[grid](a, b, c, M, N, bM, bN)

def main():
    M = 10
    N = 8

    a = torch.arange(M*N).to(device).view(M, N)
    b = torch.ones_like(a)
    c_triton = torch.empty_like(a)
    triton_add(a, b, c_triton, M, N)

    c_pytorch = a + b

    flag_match = torch.allclose(c_triton, c_pytorch, rtol = 1e-4, atol = 0)
    if(flag_match):
        print('Success')
    else:
        mismatch_idx = torch.argwhere(c_triton != c_pytorch).tolist()
        for idx in mismatch_idx:
            print(f'Mismatch at {idx[0]}, {idx[1]} | \
                {c_pytorch[idx[0],idx[1]]} != {c_triton[idx[0],idx[1]]}')
    

if __name__ == '__main__':
    main()