import os

import torch
import torch.nn as nn

import triton
import triton.language as tl

from src.utils import breakpoint_if, print_if

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['TRITON_INTERPRET']='0'

class Conv2dPytorch(nn.Module):
    def __init__(self, c_in, c_out, R, S, kernel):
        super().__init__()
        self.layers = nn.Conv2d(in_channels = c_in, out_channels = c_out, kernel_size=(R, S), bias=False)
        self.layers.weight.data = kernel

    def forward(self, x):
        B, c_in, H_in, W_in = x.shape
        return self.layers(x)

@triton.jit
def conv2d_kernel(input_features_ptr, kernel_ptr, output_features_ptr, 
    # Strides for flattened tensors (computed in Python)
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_kernel_cout, stride_kernel_cin, stride_kernel_r, stride_kernel_s,
    stride_output_b, stride_output_cout, stride_output_h, stride_output_w,
    # Dimensions
    B, c_in:tl.constexpr, H, W, c_out:tl.constexpr, R:tl.constexpr, S:tl.constexpr, 
    H_out, W_out, bm:tl.constexpr, bn:tl.constexpr, 
    R_padded:tl.constexpr, S_padded:tl.constexpr, c_out_padded:tl.constexpr, 
    B_padded:tl.constexpr, c_in_padded:tl.constexpr,
    h_out_padded:tl.constexpr, w_out_padded:tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    #Input offset and mask
    offset_1d_m = pid0*bm + tl.arange(0,bm)
    offset_1d_n = pid1*bn + tl.arange(0,bn)

    # Use padded sizes for tl.arange (power of 2 requirement)
    offset_c_in = tl.arange(0, c_in_padded)
    offset_b = tl.arange(0, B_padded)

    # Expand 1D tensors to 4D: (B, C, H, W)
    # offset_b: (B_padded,) -> (B_padded, 1, 1, 1) - batch dim
    # offset_c_in: (c_in_padded,) -> (1, c_in_padded, 1, 1) - channel dim  
    # offset_1d_m: (bm,) -> (1, 1, bm, 1) - height dim
    # offset_1d_n: (bn,) -> (1, 1, 1, bn) - width dim
    offset_b_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_b, 1), 2), -1)
    offset_c_in_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_c_in, 0), 2), -1)
    offset_m_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_1d_m, 0), 1), -1)
    offset_n_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_1d_n, 0), 1), 2)
    
    offset_input = offset_b_4d*(c_in*H*W) + offset_c_in_4d*(H*W) + \
        offset_m_4d*W + offset_n_4d


    mask_1d_m = offset_1d_m < H
    mask_1d_n = offset_1d_n < W
    mask_c_in = offset_c_in < c_in
    mask_b = offset_b < B

    # Expand masks to 4D to match offset_input shape
    mask_b_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_b, 1), 2), -1)
    mask_c_in_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_c_in, 0), 2), -1)
    mask_m_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_1d_m, 0), 1), -1)
    mask_n_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_1d_n, 0), 1), 2)
    
    mask_input = mask_b_4d & mask_c_in_4d & mask_m_4d & mask_n_4d

    #Kernel offset and mask
    # Use padded sizes (next power of 2) for tl.arange (required by Triton)
    offset_s = tl.arange(0, S_padded)
    offset_r = tl.arange(0, R_padded)
    offset_c_out = tl.arange(0, c_out_padded)

    mask_r = offset_r < R
    mask_s = offset_s < S
    mask_c_out = offset_c_out < c_out

    # Expand kernel offsets to 4D: (c_out, c_in, R, S)
    # offset_c_out: (c_out_padded,) -> (c_out_padded, 1, 1, 1)
    # offset_c_in: (c_in_padded,) -> (1, c_in_padded, 1, 1)
    # offset_r: (R_padded,) -> (1, 1, R_padded, 1)
    # offset_s: (S_padded,) -> (1, 1, 1, S_padded)
    offset_c_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_c_out, 1), 2), -1)
    offset_c_in_kernel_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_c_in, 0), 2), -1)
    offset_r_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_r, 0), 1), -1)
    offset_s_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_s, 0), 1), 2)
    
    offset_kernel = offset_c_out_4d*(c_in*R*S) + offset_c_in_kernel_4d*(R*S) + \
        offset_r_4d*S + offset_s_4d

    # Expand kernel masks to 4D
    mask_c_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_c_out, 1), 2), -1)
    mask_c_in_kernel_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_c_in, 0), 2), -1)
    mask_r_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_r, 0), 1), -1)
    mask_s_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_s, 0), 1), 2)
    
    mask_kernel = mask_c_out_4d & mask_c_in_kernel_4d & mask_r_4d & mask_s_4d 

    #Output offset and mask
    # Create output indices: (B, c_out, h_out, w_out)
    h_out_idx = tl.arange(0, h_out_padded)
    w_out_idx = tl.arange(0, w_out_padded)
    
    # Expand to 4D: (B_padded, c_out_padded, h_out_padded, w_out_padded)
    # offset_b: (B_padded,) -> (B_padded, 1, 1, 1)
    # offset_c_out: (c_out_padded,) -> (1, c_out_padded, 1, 1)
    # h_out_idx: (h_out_padded,) -> (1, 1, h_out_padded, 1)
    # w_out_idx: (w_out_padded,) -> (1, 1, 1, w_out_padded)
    offset_b_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_b, 1), 2), -1)
    offset_c_out_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_c_out, 0), 2), -1)
    h_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(h_out_idx, 0), 1), -1)
    w_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(w_out_idx, 0), 1), 2)
    
    # Compute block offsets for spatial dimensions
    block_h_start = pid0 * (bm - R + 1)
    block_w_start = pid1 * (bn - S + 1)
    
    # Output offset calculation for 4D: (B, c_out, H_out, W_out)
    # offset = b*stride_b + c_out*stride_cout + h*stride_h + w*stride_w
    offset_output_4d = offset_b_out_4d * stride_output_b + \
                       offset_c_out_out_4d * stride_output_cout + \
                       (block_h_start + h_out_4d) * stride_output_h + \
                       (block_w_start + w_out_4d) * stride_output_w
    
    # Output mask (4D)
    mask_b_out = offset_b < B
    mask_c_out_out = offset_c_out < c_out
    mask_h_out = (h_out_idx < (bm - R + 1)) & ((block_h_start + h_out_idx) < H_out)
    mask_w_out = (w_out_idx < (bn - S + 1)) & ((block_w_start + w_out_idx) < W_out)
    
    mask_b_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_b_out, 1), 2), -1)
    mask_c_out_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_c_out_out, 0), 2), -1)
    mask_h_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_h_out, 0), 1), -1)
    mask_w_out_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_w_out, 0), 1), 2)
    
    mask_output_4d = mask_b_out_4d & mask_c_out_out_4d & mask_h_out_4d & mask_w_out_4d

    #Read Inputs - Load input block and kernel once
    input_features_slice = tl.load(input_features_ptr + offset_input, mask=mask_input)
    kernel = tl.load(kernel_ptr + offset_kernel, mask=mask_kernel)
    kernel = kernel.reshape(c_out_padded, c_in_padded*R_padded*S_padded)
    output_val = tl.zeros((B_padded, c_out_padded, h_out_padded, w_out_padded), dtype=tl.float32)
    
    # Implicit GEMM: Extract patches from the loaded block and perform GEMM
    # input_features_slice shape: (B_padded, c_in_padded, bm, bn)
    # Output block size: (h_out_padded, w_out_padded) = (bm - R + 1, bn - S + 1)
    
    # Reshape input to 2D for patch extraction
    

    tl.store(output_features_ptr + offset_output_4d, output_val, mask=mask_output_4d)


def next_power_of_2(n):
    """Compute next power of 2 >= n"""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def conv2d_triton(input_features:torch.Tensor, kernel:torch.Tensor, bm:int = 16,  bn:int = 16):
    B, c_in, H, W = input_features.shape
    c_out, _, R, S = kernel.shape
    assert input_features.shape[1] == kernel.shape[1]
    
    H_out = H - R + 1
    W_out = W - S + 1

    # Compute padded sizes (next power of 2) for tl.arange
    R_padded = next_power_of_2(R)
    S_padded = next_power_of_2(S)
    c_out_padded = next_power_of_2(c_out)
    B_padded = next_power_of_2(B)
    c_in_padded = next_power_of_2(c_in)
    h_out_padded = next_power_of_2(bm - R + 1)
    w_out_padded = next_power_of_2(bn - S + 1)

    # Compute strides for flattened tensors
    # Input: (B, c_in, H, W) - already contiguous, compute strides
    stride_input_b = c_in * H * W
    stride_input_c = H * W
    stride_input_h = W
    stride_input_w = 1
    
    # Kernel: (c_out, c_in, R, S)
    stride_kernel_cout = c_in * R * S
    stride_kernel_cin = R * S
    stride_kernel_r = S
    stride_kernel_s = 1
    
    # Output: (B, c_out, H_out, W_out)
    stride_output_b = c_out * H_out * W_out
    stride_output_cout = H_out * W_out
    stride_output_h = W_out
    stride_output_w = 1

    output_features = torch.zeros((B, c_out, H_out, W_out), dtype=torch.float32, device=input_features.device)
    grid_dict = {'bm':bm, 'bn':bn}
    grid = lambda meta:(triton.cdiv(H,meta['bm']), triton.cdiv(W,meta['bn']))
    conv2d_kernel[grid(grid_dict)](
        input_features, kernel, output_features,
        stride_input_b, stride_input_c, stride_input_h, stride_input_w,
        stride_kernel_cout, stride_kernel_cin, stride_kernel_r, stride_kernel_s,
        stride_output_b, stride_output_cout, stride_output_h, stride_output_w,
        B, c_in, H, W, c_out, R, S, H_out, W_out, bm, bn, 
        R_padded, S_padded, c_out_padded, B_padded, c_in_padded,
        h_out_padded, w_out_padded)

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

    output_triton = conv2d_triton(input_image, kernel)

    print(f'Output_image : {output_pytorch}')

    print(f'Output_image : {output_triton}')

if __name__ == '__main__':
    main()