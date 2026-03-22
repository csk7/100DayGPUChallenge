import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block-aligned dispatch  (Triton kernel for scatter + vectorized PyTorch)
# ---------------------------------------------------------------------------

@triton.jit
def _dispatch_scatter_kernel(
    sort_order_ptr,
    flat_weights_ptr,
    sorted_expert_flat_ptr,
    raw_offsets_ptr,
    padded_offsets_ptr,
    out_token_ids_ptr,
    out_weights_ptr,
    num_valid,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_valid

    so = tl.load(sort_order_ptr + offs, mask=mask, other=0).to(tl.int64)
    expert = tl.load(sorted_expert_flat_ptr + offs, mask=mask, other=0).to(tl.int64)
    raw_off = tl.load(raw_offsets_ptr + expert, mask=mask, other=0).to(tl.int64)
    pad_off = tl.load(padded_offsets_ptr + expert, mask=mask, other=0).to(tl.int64)

    local_pos = offs.to(tl.int64) - raw_off
    padded_pos = pad_off + local_pos

    tl.store(out_token_ids_ptr + padded_pos, so.to(tl.int32), mask=mask)
    w = tl.load(flat_weights_ptr + so, mask=mask, other=0.0)
    tl.store(out_weights_ptr + padded_pos, w, mask=mask)


@triton.jit
def _fill_expert_ids_kernel(
    padded_offsets_ptr,
    expert_ids_ptr,
    num_experts,
    block_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    eid = tl.program_id(0)
    if eid >= num_experts:
        return
    start = tl.load(padded_offsets_ptr + eid).to(tl.int64)
    end = tl.load(padded_offsets_ptr + eid + 1).to(tl.int64)
    num_blks = (end - start) // block_size
    if num_blks == 0:
        return
    blk_start = start // block_size
    offs = tl.arange(0, BLOCK)
    mask = offs < num_blks
    tl.store(expert_ids_ptr + blk_start + offs, eid, mask=mask)


def block_aligned_dispatch(topk_ids, topk_weights, block_size, num_experts):
    """Sort tokens by expert and pad each expert's segment to a multiple of
    block_size.  Returns tensors ready for the grouped-GEMM Triton kernel.

    Uses Triton kernels for the scatter and expert-id fill -- no Python loops.
    """
    device = topk_ids.device
    T, K = topk_ids.shape
    num_valid = T * K

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)

    sort_order = flat_ids.argsort(stable=True)
    sorted_expert_flat = flat_ids[sort_order]

    expert_counts = torch.bincount(sorted_expert_flat, minlength=num_experts)
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size

    padded_offsets = F.pad(padded_counts.cumsum(0), (1, 0)).to(torch.int64)
    raw_offsets = F.pad(expert_counts.cumsum(0), (1, 0)).to(torch.int64)

    EM = int(padded_offsets[-1].item())

    sorted_token_ids = torch.full((EM,), num_valid, device=device, dtype=torch.int32)
    sorted_weights_out = torch.zeros(EM, device=device, dtype=topk_weights.dtype)

    SCATTER_BLOCK = 1024
    grid_scatter = (triton.cdiv(num_valid, SCATTER_BLOCK),)
    _dispatch_scatter_kernel[grid_scatter](
        sort_order, flat_weights, sorted_expert_flat,
        raw_offsets, padded_offsets,
        sorted_token_ids, sorted_weights_out,
        num_valid,
        BLOCK=SCATTER_BLOCK,
        num_warps=4,
    )

    num_blocks = EM // block_size
    expert_ids = torch.full((num_blocks,), -1, device=device, dtype=torch.int32)

    max_blocks_per_expert = triton.next_power_of_2(
        int(padded_counts.max().item()) // block_size + 1
    )
    max_blocks_per_expert = max(max_blocks_per_expert, 1)

    _fill_expert_ids_kernel[(num_experts,)](
        padded_offsets, expert_ids,
        num_experts,
        block_size=block_size,
        BLOCK=max_blocks_per_expert,
        num_warps=1,
    )

    return sorted_token_ids, expert_ids, sorted_weights_out, EM


# ---------------------------------------------------------------------------
# Triton: Grouped GEMM  (optional in-register sqReLU)
# ---------------------------------------------------------------------------

@triton.jit
def _grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    sorted_token_ids_ptr, expert_ids_ptr,
    N_out, K_in, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_ACT: tl.constexpr,
    INDIRECT_A: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M >= EM:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_token_id = pid_m * BLOCK_M + offs_m
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < EM,
        other=num_valid_tokens,
    ).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id < 0:
        return

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if INDIRECT_A:
        a_row = offs_token // top_k
    else:
        a_row = (pid_m * BLOCK_M + offs_m).to(tl.int64)
    a_ptrs = A_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B_ptr
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K_in, BLOCK_K)):
        k_remaining = K_in - k_step * BLOCK_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N_out), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_ACT:
        acc = tl.where(acc > 0.0, acc * acc, tl.zeros_like(acc))

    offs_cm = pid_m * BLOCK_M + offs_m
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N_out)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Triton: Grouped GEMM with weight-multiply + atomic scatter-add epilogue
# ---------------------------------------------------------------------------

@triton.jit
def _grouped_gemm_scatter_kernel(
    A_ptr, B_ptr, OUT_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, topk_weights_ptr,
    N_out, K_in, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_om, stride_on,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M >= EM:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_token_id = pid_m * BLOCK_M + offs_m
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < EM,
        other=num_valid_tokens,
    ).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id < 0:
        return

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # A is the intermediate buffer (EM, K_in) -- sequential access
    a_ptrs = (
        A_ptr
        + (pid_m * BLOCK_M + offs_m)[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        B_ptr
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K_in, BLOCK_K)):
        k_remaining = K_in - k_step * BLOCK_K
        k_mask = offs_k < k_remaining
        a_mask = token_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N_out), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    weights = tl.load(
        topk_weights_ptr + offs_token_id,
        mask=token_mask,
        other=0.0,
    )
    acc *= weights[:, None]

    orig_token = offs_token // top_k
    out_ptrs = OUT_ptr + orig_token[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = token_mask[:, None] & (offs_n[None, :] < N_out)
    tl.atomic_add(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def fused_grouped_gemm(
    A,                    # (T, d) or (EM, m) depending on indirect_a
    w_stack,              # (E, K_in, N_out) pre-transposed, bf16
    sorted_token_ids,     # (EM,) int32
    expert_ids,           # (num_blocks,) int32
    EM,
    top_k,
    num_valid_tokens,
    has_act=False,
    indirect_a=True,
    BLOCK_M=64,
    BLOCK_N=64,
    BLOCK_K=32,
):
    """Grouped GEMM via Triton with optional in-register sqReLU.

    When indirect_a=True, A is (T, K_in) and rows are loaded via
    sorted_token_ids // top_k.  When False, A is (EM, K_in) and rows
    are read sequentially.
    """
    device = A.device
    N_out = w_stack.shape[2]
    K_in = w_stack.shape[1]

    output = torch.zeros((EM, N_out), device=device, dtype=A.dtype)

    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N_out, BLOCK_N),)
    _grouped_gemm_kernel[grid](
        A, w_stack, output,
        sorted_token_ids, expert_ids,
        N_out, K_in, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        w_stack.stride(0), w_stack.stride(1), w_stack.stride(2),
        output.stride(0), output.stride(1),
        top_k=top_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        HAS_ACT=has_act,
        INDIRECT_A=indirect_a,
        num_warps=4,
        num_stages=3,
    )
    return output


def fused_grouped_gemm_scatter(
    intermediate,         # (EM, m) bf16  -- output of GEMM1
    w_stack,              # (E, K_in, N_out) pre-transposed, bf16
    sorted_token_ids,     # (EM,) int32
    expert_ids,           # (num_blocks,) int32
    sorted_weights,       # (EM,) float
    T,                    # number of original tokens
    top_k,
    EM,
    BLOCK_M=64,
    BLOCK_N=64,
    BLOCK_K=32,
):
    """Grouped GEMM2 with fused weight-multiply + atomic scatter-add."""
    device = intermediate.device
    N_out = w_stack.shape[2]
    K_in = w_stack.shape[1]
    num_valid_tokens = T * top_k

    output = torch.zeros((T, N_out), device=device, dtype=torch.float32)

    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N_out, BLOCK_N),)
    _grouped_gemm_scatter_kernel[grid](
        intermediate, w_stack, output,
        sorted_token_ids, expert_ids, sorted_weights,
        N_out, K_in, EM, num_valid_tokens,
        intermediate.stride(0), intermediate.stride(1),
        w_stack.stride(0), w_stack.stride(1), w_stack.stride(2),
        output.stride(0), output.stride(1),
        top_k=top_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output
