"""Kernel2: Block-aligned dispatch + fused GEMM1+sqReLU + fused GEMM2 with
weight-multiply + atomic scatter-add epilogue.  No separate combine step.
"""
import torch

from latent_moe_v3_original import MoE_V3_Original
from latent_moe_v3_triton_kernels import (
    block_aligned_dispatch,
    fused_grouped_gemm,
    fused_grouped_gemm_scatter,
)

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64


class MoE_V3_Kernel2(MoE_V3_Original):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("w1_t", self.w1_stack.transpose(1, 2).contiguous())
        self.register_buffer("w2_t", self.w2_stack.transpose(1, 2).contiguous())

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

        output_fp32 = fused_grouped_gemm_scatter(
            intermediate, self.w2_t,
            sorted_token_ids, expert_ids, sorted_weights,
            T, self.top_k, EM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return output_fp32.to(hidden_state.dtype)
