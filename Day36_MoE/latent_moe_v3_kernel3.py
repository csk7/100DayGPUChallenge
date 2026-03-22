"""Kernel3: Same as Kernel2 + CUDA stream overlap for shared expert.
The shared expert FFN runs concurrently with the routed experts.
"""
import torch

from latent_moe_v3_kernel2 import MoE_V3_Kernel2


class MoE_V3_Kernel3(MoE_V3_Kernel2):
    def __init__(self, config):
        super().__init__(config)
        self._shared_stream = torch.cuda.Stream()

    def forward(self, hidden_tensor):
        B, T, C = hidden_tensor.shape
        residual = hidden_tensor
        hidden_tensor = hidden_tensor.view(-1, C)

        with torch.cuda.stream(self._shared_stream):
            shared_out = self.shared_expert(residual)

        token_idx, exp_weight = self.router(hidden_tensor)
        routed_out = self.moe_fwd(hidden_tensor, token_idx, exp_weight).view(B, T, C)

        torch.cuda.current_stream().wait_stream(self._shared_stream)
        return routed_out + shared_out
