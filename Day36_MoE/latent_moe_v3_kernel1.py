"""Kernel1: Block-aligned dispatch + fused GEMM1+sqReLU (Triton) + fused GEMM2 (Triton, no activation).
Combine step (weight-multiply + index_add_) is still in PyTorch.
"""
import torch

from latent_moe_v3_original import MoE_V3_Original
from latent_moe_v3_triton_kernels import (
    block_aligned_dispatch,
    fused_grouped_gemm,
)

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64


class MoE_V3_Kernel1(MoE_V3_Original):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("w1_t", self.w1_stack.transpose(1, 2).contiguous())  # (E, d, m)
        self.register_buffer("w2_t", self.w2_stack.transpose(1, 2).contiguous())  # (E, m, d)

    def moe_fwd(self, hidden_state, expert_idx, expert_weight):
        T, d = hidden_state.shape
        num_valid_tokens = T * self.top_k

        sorted_token_ids, expert_ids, sorted_weights, EM = block_aligned_dispatch(
            expert_idx, expert_weight, BLOCK_M, self.n_experts,
        )

        intermediate = fused_grouped_gemm(
            hidden_state, self.w1_t, sorted_token_ids, expert_ids,
            EM, self.top_k, num_valid_tokens, has_act=True, indirect_a=True,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        flat_output = fused_grouped_gemm(
            intermediate, self.w2_t, sorted_token_ids, expert_ids,
            EM, self.top_k, num_valid_tokens, has_act=False, indirect_a=False,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        valid_mask = sorted_token_ids < num_valid_tokens
        valid_pos = valid_mask.nonzero(as_tuple=True)[0]
        orig_tokens = (sorted_token_ids[valid_pos].to(torch.long) // self.top_k)
        valid_out = flat_output[valid_pos] * sorted_weights[valid_pos].unsqueeze(1)

        output = torch.zeros((T, d), device=hidden_state.device, dtype=torch.float32)
        output.index_add_(0, orig_tokens, valid_out.float())
        return output.to(hidden_state.dtype)
