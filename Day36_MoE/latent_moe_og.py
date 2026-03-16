from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class NemotronRouter(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.topk = config.top_k
        self.n_groups = config.n_groups
        self.n_experts = config.n_experts
        self.topk_groups = config.topk_groups
        self.norm_weights = config.norm_weights
        self.routed_scaling_factor = config.scaling_factor

        self.weight = torch.nn.Parameter(torch.empty((config.n_experts, config.hidden_dim), dtype=torch.float32))
        self.register_buffer("scores_bias",torch.tensor([config.min_correction_val for _ in range(config.n_experts)], dtype=torch.float32))

    @torch.no_grad
    def mask_low_groups(self, score):
        #Group aware preselection
        t_tokens, n_experts = score.shape
        score += self.scores_bias.unsqueeze(0) #t_tokens, N_expert
        score = score.view(-1, self.n_groups, self.n_experts // self.n_groups)
        group_weights = torch.topk(score, k=2, dim=-1)[1].sum(-1) #(t_tokens, n_groups)
        topK_groups = torch.topk(group_weights, k=self.topk_groups, dim=-1)[0] #(t_token, topK groups)
        group_mask = torch.zeros((t_tokens, self.n_groups))
        group_mask.scatter_(1, topK_groups, 1)
        group_mask = group_mask.unsqueeze(-1).expand(t_tokens, self.n_groups, self.n_experts // self.n_groups).view(-1, self.n_experts)
        score.masked_fill(~group_mask, 0.0)
        return score
    
    def forward(self, hidden_tensor:torch.tensor) -> torch.tensor:
        hidden_tensor = hidden_tensor.type(torch.float32) @ self.weight.transpose(1,0).type(torch.float32) #(t_total, n_experts)
        hidden_tensor = F.sigmoid(hidden_tensor)
        score = self.mask_low_groups(hidden_tensor) #(t_tokens,n_experts)
        expert_idx, _ = torch.topk(score, k=self.topk, dim=-1) #(t_tokens, top_k)
        weights = torch.gather(hidden_tensor, 1, expert_idx) #(t_tokens, top_k)
        if(self.norm_weights):
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.routing_scaling_factor

        return expert_idx, weights #(t_tokens, top_k), (t_tokens, top_k)


class NemotronExpert(nn.Module):
    def __init__(self,config):
        super().__init__()

class NemotronMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList(
            [ 
                NemotronExpert(config) for _ in range(config.n_experts)
            ]
        )

class NemotronFFN_MoE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.router = NemotronRouter(config)
        self.moe = NemotronMoE(config)
        self.shared_expert = NemotronExpert(config)

    def forward(self, hidden_tensor:torch.tensor) ->torch.tensor:
        B,T,C = hidden_tensor.shape
        residual = hidden_tensor
        hidden_tensor = hidden_tensor.view(-1,C)
        token_idx, exp_weight = self.router(hidden_tensor) #(B*T, 6) ; (B*T,6)
        hidden_tensor = self.moe(hidden_tensor, token_idx, exp_weight) #()
        hidden_tensor = hidden_tensor.view(B,T,C)
        hidden_tensor = self.shared_expert(residual) + hidden_tensor
        return hidden_tensor
