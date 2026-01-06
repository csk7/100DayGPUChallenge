from ast import Dict
import os
import matplotlib

from numpy import quantile
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

@triton.jit
def gemm_kernel_swizzle(a_ptr, b_ptr, c_ptr, M, N, K, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz:tl.constexpr):
    '''Naive GeMM kernel'''
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    num_pid0, num_pid1 = tl.num_programs(0), tl.num_programs(1)

    pid0, pid1 = tl.swizzle2d(pid0, pid1, num_pid0, num_pid1, group_sz)

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

def gemm_custom(a: torch.Tensor, b: torch.Tensor, M:int, N:int, K:int, bs:int = 16, swizzle:bool = False, group_sz_num:int = 8) -> torch.Tensor:
    bm = bs
    bn = bs
    bk = bs

    group_sz = {} if swizzle is None else {'group_sz' : group_sz_num}
    grid_dict = {'bm':bm, 'bn':bn}

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (ceil_div(M,meta['bm']), ceil_div(N,meta['bn']))

    if not swizzle:
        gemm_kernel[grid(grid_dict)](a, b, c, M, N, K, bm, bn, bk)
    else:
        gemm_kernel_swizzle[grid(grid_dict)](a, b, c, M, N, K, bm, bn, bk, **group_sz)

    return c

def gemm_pytorch(a: torch.Tensor, b:torch.Tensor):
    return a @ b

def testing():
    M = 1024
    K = 512
    N = 2048

    a = torch.randn((M,K), dtype = torch.float32, device=device)
    b = torch.randn((K,N), dtype = torch.float32, device=device)
    c_pytorch = gemm_pytorch(a, b)

    
    c_triton = gemm_custom(a, b, M, N, K)

    goldens_match = torch.allclose(c_pytorch, c_triton, rtol=1e-2, atol=1e-1)
    if(goldens_match):
        print(f'Success')
    else:
        mismatch_locations = torch.argwhere(c_pytorch != c_triton).tolist()
        for idx in mismatch_locations:
            print(f'Mismatch at {idx[0]}, {idx[1]} \
                , {c_pytorch[idx[0],idx[1]]} != {c_triton[idx[0],idx[1]]}')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'], x_vals=[2**i for i in range(4,7,1)], x_log=True,
        line_arg='method', line_vals=['pytorch','naive','grouped'], line_names=['pytorch','naive','grouped'],
        styles=[('blue','-'),('green','-'),('orange','-')],
        ylabel='GB/s',
        plot_name='Triton matmul',
        args={},
    )
)
def benchmark(batch_size, method):
    a = torch.randn((1024, 1024), dtype = torch.float32, device=device)
    b = torch.randn((1024, 1024), dtype = torch.float32, device=device)
    quantiles = [0.5,0.2,0.8]
    if method == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_pytorch(a, b), quantiles = quantiles)
    if method == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_custom(a = a, b = b, M = 1024, N = 1024,
            K =1024, bs = batch_size), quantiles=quantiles)
    if method == 'grouped':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_custom(a = a, b = b, M = 1024, N = 1024,
            K =1024, bs = batch_size, swizzle=True), quantiles=quantiles)
    gbps = lambda ms: 12*1024*1024/ms * 1e-6
    return gbps(ms), gbps(min_ms), gbps(max_ms)


###Different Matrix Size Sweep###
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['matrix_size'], x_vals=[2**i for i in range(5, 12, 1)], x_log = True,
        line_arg='method', line_vals=['pytorch','naive','grouped'], line_names=['pytorch','naive','grouped'],
        styles=[('blue','-'),('green','-'),('orange','-')],
        ylabel='GB/s',
        plot_name='Triton matmul',
        args={},
    )
)
def benchmark(matrix_size, method):
    a = torch.randn((matrix_size, matrix_size), dtype = torch.float32, device=device)
    b = torch.randn((matrix_size, matrix_size), dtype = torch.float32, device=device)
    quantiles = [0.5, 0.2, 0.8]
    if method == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_pytorch(a, b), quantiles=quantiles)
    if method == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_custom(a = a, b = b, M = matrix_size,
            N=matrix_size, K=matrix_size, bs=64), quantiles=quantiles)
    if method == 'grouped':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_custom(a = a, b = b, M = matrix_size,
            N=matrix_size, K=matrix_size, bs=64, swizzle=True), quantiles=quantiles)
    gbps = lambda ms: 12*matrix_size*matrix_size/ms * 1e-6
    return gbps(ms), gbps(min_ms), gbps(max_ms)

def main():
    benchmark.run(print_data=True, show_plots=True)

if __name__ == '__main__':
    main()