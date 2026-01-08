import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_valid_nchw_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, Cin: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Cout: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    x_sb: tl.constexpr, x_sc: tl.constexpr, x_sh: tl.constexpr, x_sw: tl.constexpr,
    w_so: tl.constexpr, w_si: tl.constexpr, w_sr: tl.constexpr, w_ss: tl.constexpr,
    y_sb: tl.constexpr, y_so: tl.constexpr, y_sh: tl.constexpr, y_sw: tl.constexpr,
    BLOCK_M: tl.constexpr,  # output spatial block (OH*OW)
    BLOCK_N: tl.constexpr,  # output channels block
    BLOCK_K: tl.constexpr,  # reduction block (Cin*R*S)
):
    # 3D launch: (pid_m, pid_n, pid_b)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    M = OH * OW
    K = Cin * R * S

    # m = flattened spatial index in [0, M)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M

    # (oh, ow) from flattened m
    oh = m // OW
    ow = m - oh * OW

    # n = output channel indices in [0, Cout)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < Cout

    # accumulator: [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction loop over k in [0, K)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K

        # map k -> (cin, r, s)
        rs = k % (R * S)
        cin = k // (R * S)
        r = rs // S
        s = rs - r * S

        # input addresses for a-block: shape [BM, BK]
        # x[b, cin, oh+r, ow+s]
        ih = oh[:, None] + r[None, :]
        iw = ow[:, None] + s[None, :]

        x_ptrs = (
            x_ptr
            + pid_b * x_sb
            + cin[None, :] * x_sc
            + ih * x_sh
            + iw * x_sw
        )

        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(x_ptrs, mask=a_mask, other=0.0)

        # weight addresses for b-block: shape [BK, BN]
        # w[n, cin, r, s]  (w is (Cout,Cin,R,S))
        w_ptrs = (
            w_ptr
            + n[None, :] * w_so
            + cin[:, None] * w_si
            + r[:, None] * w_sr
            + s[:, None] * w_ss
        )
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        # dot: (BM,BK) x (BK,BN) -> (BM,BN)
        acc += tl.dot(a, b)

    # store output: y[b, n, oh, ow]
    # y strides are in elements
    y_ptrs = (
        y_ptr
        + pid_b * y_sb
        + n[None, :] * y_so
        + oh[:, None] * y_sh
        + ow[:, None] * y_sw
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv2d_valid_triton(x: torch.Tensor, w: torch.Tensor,
                        BM=128, BN=32, BK=32, num_warps=4, num_stages=2):
    """
    x: (B,Cin,H,W)  NCHW
    w: (Cout,Cin,R,S)
    returns y: (B,Cout,H-R+1,W-S+1)
    """
    assert x.ndim == 4 and w.ndim == 4
    B, Cin, H, W = x.shape
    Cout, Cin2, R, S = w.shape
    assert Cin == Cin2
    OH, OW = H - R + 1, W - S + 1
    assert OH > 0 and OW > 0

    # Triton tl.dot is best with fp16/bf16
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)
    if w.dtype not in (torch.float16, torch.bfloat16):
        w = w.to(x.dtype)

    y = torch.empty((B, Cout, OH, OW), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(OH * OW, BM), triton.cdiv(Cout, BN), B)

    conv2d_valid_nchw_kernel[grid](
        x, w, y,
        B=B, Cin=Cin, H=H, W=W,
        Cout=Cout, R=R, S=S,
        OH=OH, OW=OW,
        x_sb=x.stride(0), x_sc=x.stride(1), x_sh=x.stride(2), x_sw=x.stride(3),
        w_so=w.stride(0), w_si=w.stride(1), w_sr=w.stride(2), w_ss=w.stride(3),
        y_sb=y.stride(0), y_so=y.stride(1), y_sh=y.stride(2), y_sw=y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


# quick correctness check
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    B, Cin, H, W = 2, 8, 16, 16
    Cout, R, S = 12, 3, 3

    x = torch.randn(B, Cin, H, W, device=device, dtype=torch.float16)
    w = torch.randn(Cout, Cin, R, S, device=device, dtype=torch.float16)

    y_tri = conv2d_valid_triton(x, w)  # fp32 output
    y_ref = torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=0).to(torch.float32)

    flag_match = torch.allclose(y_tri, y_ref, rtol = 1e-4, atol = 1e-1)
    if(flag_match):
        print('Success')
    else:
        mismatch_idx = torch.argwhere(y_tri != y_ref).tolist()
        for idx in mismatch_idx:
            print(f'Mismatch at {idx[0]}, {idx[1]}, {idx[2]}, {idx[3]} | \
                {y_ref[idx[0],idx[1],idx[2],idx[3]]} != {y_tri[idx[0],idx[1],idx[2],idx[3]]}')
