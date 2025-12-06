import os
import time
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.cpp_extension import load

np.random.seed(2026)
device = 'cpu'
if(torch.cuda.is_available):
    device = 'cuda' 

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
cudaAttention = load(name = 'cudaAttention', sources=['FA_Wrapper.cpp', 'FA_Naive.cu'], extra_cuda_cflags=['-O3'])

def pytorchGoldens(Q:torch.Tensor , K:torch.Tensor, V:torch.Tensor) -> torch.Tensor:
    S = Q @ K.t() # #Nxd @ dxN ---> NxN
    P = F.softmax(S, dim = -1) #NxN
    O = P @ V #NxN @ Nxd -----> Nxd

    return O

def benchmark(func, *args, name):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputTensor = func(*args)
    end.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(end)
    print(f'{name} time taken is : {duration:.3f}ms')
    return outputTensor, duration

if __name__ == '__main__':
    #Generate random input tensors for Q, K, V
    N = 16*1024
    d = 64

    np_Q = np.int32(np.random.randn(N,d)*10)
    np_K = np.int32(np.random.randn(N,d)*10)
    np_V = np.int32(np.random.randn(N,d)*10)

    Q = torch.clip(torch.tensor(np_Q, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd
    K = torch.clip(torch.tensor(np_K, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd
    V = torch.clip(torch.tensor(np_V, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd

    #Call Pytorch Normal Attention and Flash Attention
    O_pytorch,_ = benchmark(pytorchGoldens, Q, K, V, name = 'Pytorch Naive')
    #CUDA Flash Attention Naive
    O_cuda,_ = benchmark(cudaAttention.flashAttentionNaiveLauncher,Q, K, V, name='Flash Attention Naive')
    
    #Goldens test
    tolerance = 1e-4
    allclose = torch.allclose(O_pytorch, O_cuda, rtol=0, atol=tolerance)
    if(allclose):
        print(f'Success')
    else:
        print(f'Fail')
    

