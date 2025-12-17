import os
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

torch.manual_seed(2026)

FA2custom = load(name = 'FA2custom', sources=['flashAttentionWrapper.cpp', 'flashAttention2_v1.cu'], 
                    extra_cuda_cflags=['-arch=sm_86', '-lineinfo', '--ptxas-options=-v', '-O3', '--use_fast_math'], verbose = True)

def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()

def print_mismatch(tensor1:torch.Tensor, tensor2:torch.Tensor, tolerance:float=1e-2):
    assert(tensor1.shape == tensor2.shape)

    mismatchMask = torch.abs(tensor1 - tensor2) > tolerance
    indices = torch.nonzero(mismatchMask, as_tuple=False)

    for idx in indices:
        i, j, k1, k2 = idx.tolist()
        v1 = tensor1[i, j, k1, k2].item()
        v2 = tensor2[i, j, k1, k2].item()
        print(f"Mismatch at [{i},{j},{k1},{k2}]"
                    f"Val Ref : {v1} != Val CUDA : {v2}")

def print_speedups(totalOps:float, time1: float, time2:float):

    name = "Pytorch Flash Attention"
    print(f'Time - {name} : {time1:.3f}')
    name ="Custom CUDA"
    print(f'Time - {name} : {time2:.3f}')

    reference_TFLOPS = (totalOps)/(time1 * (10**(9)))  
    print(f"Pytorch TFLOPS: {reference_TFLOPS:.3f}")
    custom_TFLOPS = (totalOps)/(time2 * (10**(9)))  
    print(f"Custom CUDA FLOPS: {custom_TFLOPS:.3f}")


def benchmark(func, *args):
    #Warm up
    for _ in range(10):
        _ = func(*args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    time_ms = []
    #Do this for N times and measure the time
    for _ in range(5):
        #Clear Cache lines
        torch.cuda.empty_cache()
        #Sync Device and time measurement
        start.record()
        output = func(*args)
        end.record()
        torch.cuda.synchronize()
        #Add all times
        time_ms.append(start.elapsed_time(end)) #ms

    #Avg Times
    avg_ms = sum(time_ms)/len(time_ms)
    #Return output and time
    return output, avg_ms

def main():
    #Generate Random inputs
    batchSize = 4
    nH = 1
    seqLength = 8192
    dim = 128
    totalOps = 2*batchSize*nH*(seqLength*seqLength*dim + seqLength*dim*seqLength)

    Q = generate_input(batchSize, nH, seqLength, dim)
    K = generate_input(batchSize, nH, seqLength, dim)
    V = generate_input(batchSize, nH, seqLength, dim)

    #Goldens
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        O_reference, pytorch_time = benchmark(F.scaled_dot_product_attention, Q, K, V)

    #Custom
    torch.cuda.synchronize()
    O_custom, custom_time = benchmark(FA2custom.sdpa_v1, Q, K, V);

    #Match
    tolerance = 1e-2
    allclose = torch.allclose(O_reference, O_custom, rtol=0, atol=tolerance)
    if(allclose):
        print(f'Success')
        print_speedups(totalOps, pytorch_time, custom_time)
    else:
        print(f'Fail')
        print_mismatch(O_reference, O_custom, tolerance)


if __name__ == "__main__":
    main()