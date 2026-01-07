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
def conv2d_kernel(input_features_ptr, kernel_ptr, output_features_ptr, B, c_in:tl.constexpr, H, W, c_out:tl.constexpr, R:tl.constexpr, S:tl.constexpr, H_out, W_out,
    bm:tl.constexpr, bn:tl.constexpr, R_padded:tl.constexpr, S_padded:tl.constexpr, c_out_padded:tl.constexpr, 
    h_out_padded:tl.constexpr, w_out_padded:tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    #Input offset and mask
    offset_1d_m = pid0*bm + tl.arange(0,bm)
    offset_1d_n = pid1*bn + tl.arange(0,bn)

    offset_c_in = tl.arange(0,1)
    offset_b = tl.arange(0,1)

    # Expand 1D tensors to 4D: (B, C, H, W)
    # offset_b: (1,) -> (1, 1, 1, 1) - batch dim
    # offset_c_in: (1,) -> (1, 1, 1, 1) - channel dim  
    # offset_1d_m: (bm,) -> (1, 1, bm, 1) - height dim
    # offset_1d_n: (bn,) -> (1, 1, 1, bn) - width dim
    offset_b_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_b, 0), 1), 2)
    offset_c_in_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_c_in, 0), 1), 2)
    offset_m_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_1d_m, 0), 1), -1)
    offset_n_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(offset_1d_n, 0), 1), 2)
    
    offset_input = offset_b_4d*(c_in*H*W) + offset_c_in_4d*(H*W) + \
        offset_m_4d*W + offset_n_4d


    mask_1d_m = offset_1d_m < H
    mask_1d_n = offset_1d_n < W
    mask_c_in = offset_c_in < c_in
    mask_b = offset_b < B

    # Expand masks to 4D to match offset_input shape
    mask_b_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_b, 0), 1), 2)
    mask_c_in_4d = tl.expand_dims(tl.expand_dims(tl.expand_dims(mask_c_in, 0), 1), 2)
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
    # offset_c_out: (c_out,) -> (c_out, 1, 1, 1)
    # offset_c_in: (1,) -> (1, 1, 1, 1)
    # offset_r: (R,) -> (1, 1, R, 1)
    # offset_s: (S,) -> (1, 1, 1, S)
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


    #Read Inputs
    input_features_slice = tl.load(input_features_ptr + offset_input, mask=mask_input)
    kernel = tl.load(kernel_ptr + offset_kernel, mask=mask_kernel)

    # Compute convolution directly
    # Initialize output accumulator by loading zeros or computing directly
    # Workaround: Instead of tl.zeros, initialize by computation
    # Create indices for output block
    h_out_idx = tl.arange(0, h_out_padded)
    w_out_idx = tl.arange(0, w_out_padded)
    
    # Initialize accumulator to zero
    # Workaround: Use arange and multiply by 0.0 instead of tl.zeros
    # (tl.zeros has issues with constexpr parameters in interpreter mode)
    ones_h = tl.arange(0, h_out_padded).to(tl.float32) * 0.0 + 1.0
    ones_w = tl.arange(0, w_out_padded).to(tl.float32) * 0.0 + 1.0
    output_val = tl.expand_dims(ones_h, 1) + tl.expand_dims(ones_w, 0)
    
    # TODO: Implement proper convolution computation
    # For now, this initializes output to zero
    # The actual convolution needs to:
    # 1. Iterate over output positions (h_out, w_out) in the block
    # 2. For each position, iterate over kernel (r, s) and input channels (c_in)
    # 3. Accumulate: output[h_out, w_out] += 
    #    input[c_in, h_out+r, w_out+s] * kernel[c_out, c_in, r, s]
    # 4. Handle multiple output channels (c_out) - may need outer loop

    #Write result  
    # Calculate 1D offset for output block
    output_offset_2d = tl.expand_dims(h_out_idx, 1) * W_out + tl.expand_dims(w_out_idx, 0)
    
    # Compute block offset for this program ID
    block_h_offset = pid0 * (bm-R+1)
    block_w_offset = pid1 * (bn-S+1)
    block_offset = block_h_offset * W_out + block_w_offset
    
    # Store output with proper masking
    output_ptr_2d = output_features_ptr + block_offset + output_offset_2d
    output_mask_2d = (tl.expand_dims(h_out_idx, 1) < (bm-R+1)) & (tl.expand_dims(w_out_idx, 0) < (bn-S+1))
    tl.store(output_ptr_2d, output_val, mask=output_mask_2d)


def next_power_of_2(n):
    """Compute next power of 2 >= n"""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def conv2d_triton(input_features:torch.Tensor, kernel:torch.Tensor, bm:int = 4,  bn:int = 4):
    B, c_in, H, W = input_features.shape
    c_out, _, R, S = kernel.shape
    assert input_features.shape[1] == kernel.shape[1]
    
    H_out = H - R + 1
    W_out = W - S + 1

    # Compute padded sizes (next power of 2) for tl.arange
    R_padded = next_power_of_2(R)
    S_padded = next_power_of_2(S)
    c_out_padded = next_power_of_2(c_out)
    h_out_padded = next_power_of_2(bm - R + 1)
    w_out_padded = next_power_of_2(bn - S + 1)

    output_features = torch.zeros((c_out, H_out, W_out), dtype=torch.float32, device='cuda')
    grid_dict = {'bm':bm, 'bn':bn}
    grid = lambda meta:(triton.cdiv(H,meta['bm']), triton.cdiv(W,meta['bn']))
    conv2d_kernel[grid(grid_dict)](input_features, kernel, output_features, B, c_in, H, W,
        c_out, R, S, H_out, W_out, bm, bn, R_padded, S_padded, c_out_padded, h_out_padded, w_out_padded)
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

    output_triton = conv2d_triton(input_image, kernel)

    print(f'Output_image : {output_pytorch}')

    print(f'Output_image : {output_triton}')

if __name__ == '__main__':
    main()