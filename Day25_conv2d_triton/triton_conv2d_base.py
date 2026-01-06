import os

import torch
import torch.nn as nn

import triton
import triton.language as tl

from src.utils import breakpoint_if, print_if

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['TRITON_INTERPRET']='1'

class Conv2dPytorch(nn.Module):
    def __init__(self, c_in, c_out, R, S, kernel):
        super().__init__()
        self.layers = nn.Conv2d(in_channels = c_in, out_channels = c_out, kernel_size=(R, S), bias=False)
        self.layers.weight.data = kernel

    def forward(self, x):
        B, c_in, H_in, W_in = x.shape
        return self.layers(x)

@triton.jit
def conv2d_kernel(input_features_ptr, kernel_ptr, output_features_ptr, B, c_in, H, W, c_out, R, S, H_out, W_out,
    bm:tl.constexpr, bn:tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    #Input offset and mask
    offset_1d_m = pid0*bm + tl.arange(0,bm)
    offset_1d_n = pid1*bn + tl.arange(0,bn)

    offset_c_in = tl.arange(0,1)
    offset_b = tl.arange(0,1)

    offset_input = tl.expand_dims(offset_b,3)*(c_in*H*W) + tl.expand_dims(offset_c_in,2)*(H*W) + \
        tl.expand_dims(offset_1d_m, 1)*W + tl.expand_dims(offset_1d_n,0)


    mask_1d_m = offset_1d_m < H
    mask_1d_n = offset_1d_n < W
    mask_c_in = offset_c_in < c_in
    mask_b = offset_b < B

    mask_input = tl.expand_dims(mask_b,3) & tl.expand_dims(mask_c_in,2) &\
        tl.expand_dims(mask_1d_m,1) & tl.expand_dims(mask_1d_n, 0)

    #Kernel offset and mask
    offset_s = tl.arange(0, S)
    offset_r = tl.arange(0, R)

    offset_c_out = tl.arange(0, c_out)

    mask_r = offset_r < R
    mask_s = offset_s < S
    mask_c_out = offset_c_out < c_out

    offset_kernel = tl.expand_dims(offset_c_out,3)*(c_in*R*S) + tl.expand_dims(offset_c_in,2)*(R*S) + \
        tl.expand_dims(offset_r, 1)*S + tl.expand_dims(offset_s,0)

    mask_kernel = tl.expand_dims(mask_c_out,3)*c_in*R*S + tl.expand_dims(mask_c_in,2)*R*S +\
        tl.expand_dims(mask_r,1)*S + tl.expand_dims(mask_s,0) 

    #Output offset and mask
    offset_h_out = pid0*(bm-R+1) + tl.arange(bm-R+1)
    offset_w_out = pid1*(bn-S+1) + tl.arange(bn-S+1) 

    mask_h_out = offset_h_out < H_out
    mask_w_out = offset_w_out < W_out

    offset_output = tl.expand_dims(offset_b,3)*(c_out*H*W) + tl.expand_dims(offset_c_out,2)*(H_out*W_out) + \
        tl.expand_dims(offset_h_out, 1)*W_out + tl.expand_dims(offset_w_out,0)

    mask_output = tl.expand_dims(mask_b,3) & tl.expand_dims(mask_c_out,2) &\
        tl.expand_dims(mask_h_out,1) & tl.expand_dims(mask_w_out, 0)

    #Read Inputs
    input_features_slice = tl.load(input_features_ptr + offset_input, mask=mask_input)
    kernel = tl.load(kernel_ptr + offset_kernel, mask=mask_kernel)

    #Convert to 2d vals
    #Features is B,Cin,bm,bn ; Kernels Cout, Cin, R, S
    temp_input = tl.zeros(B,((bm-R+1)*(bn-S+1)), c_in*R*S, dtype=tl.float32, device = 'cuda')
    for idx_row in range(bm-R+1):
        for idx_col in range(bn-S+1):
            temp_input[:,idx_row*(bn-S+1) + idx_col,:] = \
                input_features_slice[:,:,idx_row:idx_row+R:idx_col+idx_col+S].reshape(B,c_in*R*S).unsqueeze(1)
    temp_input = temp_input.reshape(B*((bm-R+1)*(bn-S+1)), c_in*R*S)
    kernel  =  tl.trans(kernel.reshape(c_out,c_in*R*S))
    
    #Matmul
    output_val = tl.dot(temp_input, kernel) #(B*bm_out*bn_out, cout)

    #Convert back to correct shape
    output_val = output_val.reshape(B, (bm-R+1),(bn-S+1), c_out)
    output_val = tl.trans(output_val, 0, 3, 1, 2)

    #Write result
    tl.store(output_features_ptr + offset_output, output_val, mask_output)


def conv2d_triton(input_features:torch.Tensor, kernel:torch.Tensor, bm:int = 4,  bn:int = 4):
    B, c_in, H, W = input_features.shape
    c_out, _, R, S = kernel.shape
    assert input_features.shape[1] == kernel.shape[1]
    
    H_out = H - R + 1
    W_out = W - S + 1

    output_features = torch.zeros((c_out, H_out, W_out), dtype=torch.float32, device='cuda')
    grid_dict = {'bm':bm, 'bn':bn}
    grid = lambda meta:(triton.cdiv(H,meta['bm']), triton.cdiv(W,meta['bn']))
    conv2d_kernel[grid(grid_dict)](input_features, kernel, output_features, B, c_in, H, W,
        c_out, R, S, H_out, W_out, bm, bn)
    output_features.view(c_out, H_out, W_out)

    return output_features



def main():
    H = 4
    W = 4

    R = 3
    S = 3

    input_image = torch.arange(H*W, dtype = torch.float32, device=device).view(H, W).unsqueeze(0).unsqueeze(0)
    kernel = torch.arange(R*S, dtype = torch.float32, device=device).view(1, 1 ,R, S)

    model = Conv2dPytorch(1, 1, R, S, kernel)

    with torch.no_grad():
        model.to(device)

    output_pytorch = model(input_image)

    print(f'Output_image : {output_pytorch}')

if __name__ == '__main__':
    main()