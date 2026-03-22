import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_weights_from_file(file_name:str):
    dir_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'nemotron_nano')
    file = os.path.join(dir_location, f"{file_name}.pt")
    data = torch.load(file, weights_only=True)
    weights, bias = data['weight'], data['bias']
    return weights, bias

class NemotronRouter(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.topk = config.top_k
        self.n_groups = config.n_groups
        self.n_experts = config.n_experts
        self.topk_groups = config.topk_groups
        self.norm_weights = config.norm_weights
        self.routing_scaling_factor = config.scaling_factor

        router_w, router_b = read_weights_from_file("router")
        self.weight = torch.nn.Parameter(router_w.to(torch.float32))
        self.register_buffer("router_bias", router_b.to(torch.float32))
        self.register_buffer("scores_bias",torch.zeros(config.n_experts, dtype=torch.float32))

    @torch.no_grad()
    def mask_low_groups(self, score_init):
        #Group aware preselection
        t_tokens, n_experts = score_init.shape
        score_init += self.scores_bias.unsqueeze(0) #t_tokens, N_expert
        score = score_init.view(-1, self.n_groups, self.n_experts // self.n_groups)
        group_weights = torch.topk(score, k=2, dim=-1)[0].sum(dim=-1) #(t_tokens, n_groups)
        topK_groups = torch.topk(group_weights, k=self.topk_groups, dim=-1)[1] #(t_token, topK groups)
        group_mask = torch.zeros_like(group_weights)
        group_mask.scatter_(1, topK_groups, 1)
        group_mask = group_mask.unsqueeze(-1).expand(t_tokens, self.n_groups, self.n_experts // self.n_groups).reshape(-1, self.n_experts)
        score_init = score_init.masked_fill(~group_mask.bool(), 0.0)
        result_idx = torch.topk(score_init, k=self.topk, dim=-1)[1] #(t_tokens, top_k)
        return result_idx
    
    def forward(self, hidden_tensor:torch.tensor) -> torch.tensor:
        hidden_tensor = hidden_tensor.type(torch.float32) @ self.weight.transpose(1,0).type(torch.float32) + self.router_bias #(t_total, n_experts)
        hidden_tensor = F.sigmoid(hidden_tensor)
        expert_idx = self.mask_low_groups(hidden_tensor) #(t_tokens,n_experts)
        weights = torch.gather(hidden_tensor, 1, expert_idx) #(t_tokens, top_k)
        if(self.norm_weights):
            weights /= (weights.sum(dim=-1, keepdim=True) + 1e-20)
        
        weights *= self.routing_scaling_factor

        return expert_idx, weights #(t_tokens, top_k), (t_tokens, top_k)


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
        
class NemotronExpertShared(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ff1 = nn.Linear(config.d, config.moe_shared_expert_intermediate_size, bias=False)
        self.ff2 = nn.Linear(config.moe_shared_expert_intermediate_size, config.d, bias=False)
        w0, _ = read_weights_from_file("layer0_ffn_sh_0_0")
        w1, _ = read_weights_from_file("layer0_ffn_sh_0_1")
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
    
    def moe_fwd(self, hidden_state:torch.tensor, expert_idx:torch.tensor, expert_weight:torch.tensor) -> torch.tensor:
        t_total, C = hidden_state.shape
        _, K = expert_idx.shape

        final_hidden = torch.zeros_like(hidden_state, dtype=expert_weight.dtype)
        expert_mask = F.one_hot(expert_idx, num_classes=self.n_experts).permute(2,0,1) #(t_tokens, K) --> (t_tokens, K, N) --> (N, t_tokens, K)
        for exp_i, expert in zip(range(self.n_experts),self.moe):
            current_exp_mask = expert_mask[exp_i]
            token_indices, weight_indices = torch.where(current_exp_mask)

            if(token_indices.numel() > 0):
                expert_input = hidden_state[token_indices] #(t_exp,C)
                expert_out = expert(expert_input)
                weights = expert_weight[token_indices, weight_indices]
                expert_out *= weights.unsqueeze(1)
                final_hidden.index_add_(0,token_indices,expert_out)
            else:
                expert_input = torch.zeros_like(hidden_state[0]).unsqueeze(0).to(expert.ff2.weight.dtype)
                expert_out = expert(expert_input)
                final_hidden += expert_out

        return final_hidden.type(hidden_state.dtype)


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