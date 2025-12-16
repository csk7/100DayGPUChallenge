import os
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

torch.manual_seed(2026)

def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()

FA2custom = load(name = 'FA2custom', sources=['FA2Wrapper.cpp', 'cudaFA2ptx.cu'], 
                    extra_cuda_cflags=['-arch=sm_86', '-lineinfo', '--ptxas-options=-v'], verbose = True)

def main():
    #Generate Random inputs
    batchSize = 4
    nH = 1
    seqLength = 8192
    dim = 64
    scaling_const = dim ** (-0.5)

    Q = generate_input(batchSize, nH, seqLength, dim)
    K = generate_input(batchSize, nH, seqLength, dim)
    V = generate_input(batchSize, nH, seqLength, dim)
    mul1 = (Q[0,0,:,:] @ K[0,0,:,:].T)*scaling_const
    mul1_max = torch.max(mul1, dim=-1).values
    for i in range(16):
        mul1[i,:] -= mul1_max[i]
    valTemp = torch.sum(mul1,dim=-1)
    print(f'\n{valTemp}\n')

    #Goldens
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        O_reference = F.scaled_dot_product_attention(Q, K, V)

    print(f'Orefer size : {O_reference.size()}, Val : {O_reference[0,0,0,:16]}')

    #Custom
    O_custom = FA2custom.sdpa_v1(Q, K, V);

    print(f'Ocustom size : {O_custom.size()}, Val : {O_custom[0,0,0,:16]}')

    #Match
    tolerance = 1e-2
    allclose = torch.allclose(O_reference, O_custom, rtol=0, atol=tolerance)
    if(allclose):
        print(f'Success');
    else:
        print(f'Fail')
        mask = torch.abs(O_reference - O_custom) > tolerance
        for i in range(seqLength):
            for j in range(dim):
                if(mask[0,0,i,j]):
                    print(f'Error at {i,j} , {O_reference[0, 0, i, j]} != {O_custom[0, 0, i, j]}')


if __name__ == "__main__":
    main()