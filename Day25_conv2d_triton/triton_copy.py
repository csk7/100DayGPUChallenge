import os
import torch
import triton
import triton.language as tl
from src.utils import test_pid_conds, breakpoint_if, print_if, ceil_div

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['TRITON_INTERPRET']='1'

@triton.jit
def copy_triton_launch(a_ptr, b_ptr, batch_size: tl.constexpr, N: tl.constexpr):
    pid0 = tl.program_id(0)
    offset = pid0*batch_size + tl.arange(0,batch_size);
    mask = offset < N

    a_values = tl.load(a_ptr + offset, mask)
    print_if(f'pid = {pid0} | offs = {offset}, mask = {mask}, x = {a_values}', '')
    tl.store(b_ptr + offset, a_values, mask)


def main():
    N = 10
    a = torch.arange(N).to('cuda')

    b = torch.empty_like(a)
    batch_size = 4
    grid = (ceil_div(N,batch_size), 1, 1)
    copy_triton_launch[grid](a, b, batch_size, N)

    equal_flag = torch.allclose(a, b, atol = 1e-4, rtol= 0.0)

    if(equal_flag):
        print(f'Success')
        print(f'{b}')
    else:
        mismatch = torch.argwhere(a != b).tolist()
        for idx in mismatch:
            print(f'Mismatch at {idx[0]},{idx[1]} , \
                {a[idx[0], idx[1]]} != {b[idx[0], idx[1]]}')

if __name__ == '__main__':
    main()