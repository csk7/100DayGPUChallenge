import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from latent_moe_og import read_weights_from_file, NemotronRouter, NemotronExpertShared

class NemotronExpert(nn.Module):
    def __init__(self,config, expert_idx:int):
        super().__init__()
        self.ff1 = nn.Linear(config.d, config.m, bias=False)
        self.ff2 = nn.Linear(config.m, config.d, bias=False)
        w0, _ = read_weights_from_file(f"layer0_ffn_exp_{expert_idx}_0")
        w1, _ = read_weights_from_file(f"layer0_ffn_exp_{expert_idx}_1")
        with torch.no_grad():
            self.ff1.weight.copy_(w0.to(self.ff1.weight.dtype))
            self.ff2.weight.copy_(w1.to(self.ff2.weight.dtype))

    def forward(self, input_tensor:torch.tensor) -> torch.tensor:
        inter_out = self.ff1(input_tensor)
        inter_out = torch.pow(F.relu(inter_out),2)
        inter_out = self.ff2(inter_out)

        return inter_out
        
class NemotronFFN_MoE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_experts = config.n_experts

        self.router = NemotronRouter(config)
        self.moe = nn.ModuleList(
            [ 
                NemotronExpert(config, i) for i in range(config.n_experts)
            ]
        )
        self.shared_expert = NemotronExpertShared(config)

        w1_weight = [self.moe[i].ff1.weight.data for i in range(self.n_experts)]
        w2_weight = [self.moe[i].ff2.weight.data for i in range(self.n_experts)]
        self.register_buffer("w1_stack",torch.stack(w1_weight, dim=0))
        self.register_buffer("w2_stack",torch.stack(w2_weight, dim=0))
    
    def moe_fwd(self, hidden_state:torch.tensor, expert_idx:torch.tensor, expert_weight:torch.tensor) -> torch.tensor:
        #---------Dispatch---------#
        t_tokens, d = hidden_state.shape
        _, k = expert_idx.shape
        flat_expert_ids = expert_idx.flatten()
        sort_order = flat_expert_ids.argsort(stable=True)

        sorted_expert_ids = flat_expert_ids[sort_order].to(torch.long)
        flat_token_ids = (
            torch.arange(t_tokens, device=hidden_state.device)
            .unsqueeze(1)
            .expand(-1, k)
            .flatten()
        )
        sorted_token_ids = flat_token_ids[sort_order].to(torch.long)
        sorted_weights = expert_weight.flatten()[sort_order]

        flat_expert_tokens = hidden_state[sorted_token_ids]
        expert_bin_count = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        expert_offsets = F.pad(torch.cumsum(expert_bin_count, dim=0), (1, 0))
        cnt_max = int(expert_bin_count.max().item())

        #---------Scatter---------#
        local_pos = torch.arange(t_tokens * k, device=hidden_state.device, dtype=expert_offsets.dtype)
        local_pos = (local_pos - expert_offsets[sorted_expert_ids]).to(torch.long) #(T*K,)
        batched_input = torch.zeros(
            (self.n_experts, cnt_max, d),
            dtype=hidden_state.dtype,
            device=hidden_state.device,
        )
        batched_input[sorted_expert_ids, local_pos] = flat_expert_tokens

        #---------FFN---------#
        inter_tensor = torch.bmm(batched_input, self.w1_stack.transpose(1,2))
        inter_tensor = F.relu(inter_tensor).pow(2)
        batched_output = torch.bmm(inter_tensor, self.w2_stack.transpose(1,2)) #(N, max_cnt, d)

        #---------Collect and Index add---------#
        flat_hidden_output = batched_output[sorted_expert_ids, local_pos] #(T*K, d)
        flat_hidden_output *= sorted_weights.unsqueeze(1)

        final_output = torch.zeros_like(hidden_state)
        final_output.index_add_(0, sorted_token_ids, flat_hidden_output)

        return final_output.to(hidden_state.dtype)

    def forward(self, hidden_tensor:torch.tensor) ->torch.tensor:
        B,T,C = hidden_tensor.shape
        residual = hidden_tensor
        hidden_tensor = hidden_tensor.view(-1,C)
        token_idx, exp_weight = self.router(hidden_tensor) #(B*T, 6) ; (B*T,6)
        hidden_tensor = self.moe_fwd(hidden_tensor, token_idx, exp_weight).view(B,T,C) #(t_total, C)
        hidden_tensor = self.shared_expert(residual) + hidden_tensor
        return hidden_tensor


if __name__ == '__main__':
    class Config:
        def __init__(self):
            # MoE-related fields from Nemotron config.json
            self.hidden_dim = 2688
            self.n_experts = 128
            self.n_shared_experts = 1
            self.top_k = 6
            self.n_groups = 8
            self.topk_groups = 1
            self.norm_weights = True
            self.scaling_factor = 2.5
            self.mlp_bias = False
            self.mlp_hidden_act = "relu2"

            # Routed and shared expert FFN sizes
            self.moe_intermediate_size = 1856
            self.moe_shared_expert_intermediate_size = 3712

            # Fields currently consumed by this implementation
            self.d = self.hidden_dim
            self.m = self.moe_intermediate_size

    cfg = Config()
    model = NemotronFFN_MoE(cfg)
    model.to('cuda')
    x = torch.randn(1, 32, cfg.hidden_dim).to('cuda')
    y = model(x)
    print(y.shape)