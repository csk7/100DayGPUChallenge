import os
import time

import numpy as np

import torch
from torch.cpu import synchronize
from torch.cuda import ipc_collect
import torch.nn.functional as F

torch.manual_seed(2055)

def generateRandom(size:tuple, minVal:int, maxVal:int):
    return np.clip(np.random.normal(0, 1, size), minVal, maxVal) #returns a numpy array of shape [M,N] with random values taken from an uniform distribution of mean 0 and Var 1

def softmaxPytorchBenchmark():
    os.makedirs('benchmarks', exist_ok = True)
    n_iters = 5
    N_min = 32*1024
    N_max = 262144
    N = N_min

    with open('benchmarks/torch_logs.txt', 'w') as fileWrite:
        #generate random data
        M = 2048
        while(N<N_max):
            inputArray = torch.tensor(generateRandom(size = (M, N), minVal=-1, maxVal=1), dtype = torch.float32, device = "cuda")
            torch.cuda.synchronize()
            
            #Warm up
            for _ in range(5):
                _ = F.softmax(input=inputArray, dim = -1)
                torch.cuda.synchronize()

            total_time = 0;
            for i in range(n_iters):
                torch.cuda.synchronize()
                start = time.time()
                _ = F.softmax(input=inputArray, dim=-1)
                torch.cuda.synchronize()
                end = time.time()

                total_time += (end - start)*1000

            total_time /= n_iters
            print(f'Pytorch Execution Time for {M} x {N} Softmax is {total_time:.3f} ms')
            fileWrite.write(f'{M}x{N} takes {total_time} ms')
            
            #Clear state
            del inputArray
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

            N*=2
            time.sleep(1)


if __name__ == '__main__':
    softmaxPytorchBenchmark()