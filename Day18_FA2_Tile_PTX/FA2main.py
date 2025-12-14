import os
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

torch.manual_seed(2026)
device = torch.cuda()

def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()

FA2custom = load(name = 'customAttention_v1', sources=['FA2Wrapper.cpp', 'cudaFA2ptx.cu'], 
                    extra_cuda_cflags=['-lineinfo', '--ptxas-options=-v', '-O3'], verbose = True)

def main():
    #Generate Random inputs
    batchSize = 1
    seqLength = 128
    dim = 128

    Q = generate_input(batchSize, seqLength, dim)
    K = generate_input(batchSize, seqLength, dim)
    V = generate_input(batchSize, seqLength, dim)

    #Goldens
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        O_reference = F.scaled_dot_product_attention(Q, K, V)

    #Custom
    O_custom = FA2custom.sdpa(Q, K, V);

    #Match
    tolerance = 1e-4
    allclose = torch.allclose(O_reference, O_custom, rtol=0, atol=tolerance)
    if(allclose):
        print(f'Success');
    else:
        print(f'Fail')


if __name__ == "__main__":
    main()