import torch
import torch.nn as nn
import torch.nn.functional as F
from latent_moe_og import read_weights_from_file, NemotronRouter, NemotronExpertShared


class MoE_V3_Original(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.router = NemotronRouter(config)
        self.shared_expert = NemotronExpertShared(config)

        w1_list, w2_list = [], []
        for i in range(config.n_experts):
            w0, _ = read_weights_from_file(f"layer0_ffn_exp_{i}_0")
            w1, _ = read_weights_from_file(f"layer0_ffn_exp_{i}_1")
            w1_list.append(w0)
            w2_list.append(w1)
        self.register_buffer("w1_stack", torch.stack(w1_list, dim=0))  # (N, m, d)
        self.register_buffer("w2_stack", torch.stack(w2_list, dim=0))  # (N, d, m)

    def moe_fwd(self, hidden_state, expert_idx, expert_weight):
        t_tokens, d = hidden_state.shape
        _, k = expert_idx.shape

        flat_expert_ids = expert_idx.flatten()
        sort_order = flat_expert_ids.argsort(stable=True)

        sorted_expert_ids = flat_expert_ids[sort_order].to(torch.long)
        flat_token_ids = (
            torch.arange(t_tokens, device=hidden_state.device)
            .unsqueeze(1).expand(-1, k).flatten()
        )
        sorted_token_ids = flat_token_ids[sort_order].to(torch.long)
        sorted_weights = expert_weight.flatten()[sort_order]

        flat_expert_tokens = hidden_state[sorted_token_ids]
        expert_bin_count = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        expert_offsets = F.pad(torch.cumsum(expert_bin_count, dim=0), (1, 0))
        cnt_max = int(expert_bin_count.max().item())

        local_pos = torch.arange(t_tokens * k, device=hidden_state.device, dtype=expert_offsets.dtype)
        local_pos = (local_pos - expert_offsets[sorted_expert_ids]).to(torch.long)
        batched_input = torch.zeros(
            (self.n_experts, cnt_max, d),
            dtype=hidden_state.dtype, device=hidden_state.device,
        )
        batched_input[sorted_expert_ids, local_pos] = flat_expert_tokens

        inter_tensor = torch.bmm(batched_input, self.w1_stack.transpose(1, 2))
        inter_tensor = F.relu(inter_tensor).pow(2)
        batched_output = torch.bmm(inter_tensor, self.w2_stack.transpose(1, 2))

        flat_hidden_output = batched_output[sorted_expert_ids, local_pos]
        flat_hidden_output *= sorted_weights.unsqueeze(1)

        final_output = torch.zeros_like(hidden_state)
        final_output.index_add_(0, sorted_token_ids, flat_hidden_output)
        return final_output.to(hidden_state.dtype)

    def forward(self, hidden_tensor):
        B, T, C = hidden_tensor.shape
        residual = hidden_tensor
        hidden_tensor = hidden_tensor.view(-1, C)
        token_idx, exp_weight = self.router(hidden_tensor)
        hidden_tensor = self.moe_fwd(hidden_tensor, token_idx, exp_weight).view(B, T, C)
        hidden_tensor = self.shared_expert(residual) + hidden_tensor
        return hidden_tensor
