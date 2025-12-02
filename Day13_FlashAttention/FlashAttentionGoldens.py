import torch
import torch.nn.functional as F
import numpy as np

np.random.seed(2026)
device = 'cpu'
if(torch.cuda.is_available):
    device = 'cuda' 


def pytorchGoldens(Q:torch.Tensor , K:torch.Tensor, V:torch.Tensor) -> torch.Tensor:
    S = Q @ K.t() # #Nxd @ dxN ---> NxN
    print(f'S : {S.type(torch.int32)}')
    P = F.softmax(S, dim = -1) #NxN
    print(f'P : {P}')
    O = P @ V #NxN @ Nxd -----> Nxd

    return O


if __name__ == '__main__':
    #Generate random input tensors for Q, K, V
    N = 4
    d = 4
    Br = 2
    Bc = 2

    np_Q = np.int32(np.random.randn(N,d)*10)
    np_K = np.int32(np.random.randn(N,d)*10)
    np_V = np.int32(np.random.randn(N,d)*10)

    Q = torch.clip(torch.tensor(np_Q, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd
    K = torch.clip(torch.tensor(np_K, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd
    V = torch.clip(torch.tensor(np_V, device = device, dtype = torch.float32), -10.0, 10.0) #Nxd

    print(f'Q : {Q.type(torch.int32)}')
    print(f'K : {K.type(torch.int32)}')
    print(f'V : {V.type(torch.int32)}')


    #Call transforms attention library
    O_pytorch = pytorchGoldens(Q = Q, K = K, V = V)
    
    print(f'Output : {O_pytorch}')
    

