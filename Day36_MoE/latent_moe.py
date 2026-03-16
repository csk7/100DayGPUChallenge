#MoE Pytorch v1
import os
import torch
import torch.nn.functional as F

def read_weights_from_file(file_name:str):
    dir_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'nemotron_nano')
    file = os.path.join(dir_location, f"{file_name}.pt")
    data = torch.load(file, weights_only=True)
    weights, bias = data['weight'], data['bias']
    return weights, bias

def linear(act_tensor:torch.tensor, layer_name:str, device="cuda", output_dtype=torch.bfloat16) -> torch.tensor:
    weights, bias = read_weights_from_file(layer_name)
    weights, bias = weights.to(device), bias.to(device)
    out_activation = act_tensor @ weights.transpose(1,0) + bias
    out_activation = out_activation.to(output_dtype)
    return out_activation

def expert_ffn_fwd(input_tensor:torch.tensor, weight_name:str) -> torch.tensor:
    #ffn 
    intermediate_act = linear(input_tensor, f"{weight_name}_{0}") #(T_exp,d) @ (d,m) --> (T_exp, m) 
    #squared relu
    intermediate_act = torch.pow(F.relu(intermediate_act),2)
    #ffn
    output_tensor = linear(intermediate_act, f"{weight_name}_{1}") #(T_exp,m) @ (m, d) --> (T_exp, d) 

    return output_tensor


def main(B:int = 1, T:int = 32, d:int = 2688, m:int = 2, K:int = 6, N:int = 128, S:int=1, device:str = "cuda", dtype_bitwdth:int = 16, seed=42) -> torch.tensor:
    input_dtype = torch.bfloat16 if dtype_bitwdth == 16 else torch.float32
    generator = torch.Generator(device=device).manual_seed(seed)
    input_tensor = torch.randn((B,T,d), device = device, dtype = input_dtype, generator=generator)
    
    input_tensor = input_tensor.view(B*T,d)
    
    #router - Sigmoid Gating
    router_prob = linear(input_tensor,'router') #(B*T, d) @ (d,N) --> (B*T, N)
    router_prob = 1/(1+torch.exp(-1*router_prob))

    values_topK, indices_topK = torch.topk(router_prob, k=K, dim=-1)

    #Make New Tensors - All Gather
    expert_input_tensor = [torch.empty((0, d), device=device, dtype = input_dtype) for _ in range(N)]
    expert_weight_tensor = [torch.empty(0, device=device, dtype = values_topK.dtype) for _ in range(N)]
    expert_idx_tensor = [torch.empty(0, device=device, dtype=torch.int64) for _ in range(N)]
    
    #Renormalize weights
    for i in range(values_topK.shape[0]):
        values_topK[i,:] /= values_topK[i,:].sum()

    for i in range(indices_topK.shape[0]):
        for j in range(K):
            #TODO:Implement capacity policy 
            cur_expert_idx = indices_topK[i,j].item()
            expert_input_tensor[cur_expert_idx] = torch.cat([expert_input_tensor[cur_expert_idx], input_tensor[i,:].view(1,d)], dim = 0)
            expert_weight_tensor[cur_expert_idx] = torch.cat([expert_weight_tensor[cur_expert_idx], values_topK[i,j].detach().clone().view(1)], dim=0)
            expert_idx_tensor[cur_expert_idx] = torch.cat([expert_idx_tensor[cur_expert_idx], torch.tensor(i, device=device).view(1)], dim=0)

    #Expert FFN
    expert_output_tensor = []
    for i in range(N):
        t_exp, _ = expert_input_tensor[i].shape
        if(t_exp > 0):
            weight_name = f"layer0_ffn_exp_{i}"
            expert_output_tensor.append(expert_ffn_fwd(expert_input_tensor[i], weight_name))
        else:
            expert_output_tensor.append(None)

    for i in range(N):
        if expert_output_tensor[i] is not None:
            t_exp, _ = expert_output_tensor[i].shape
            if(t_exp > 0):
                expert_output_tensor[i] *= expert_weight_tensor[i].unsqueeze(dim=-1)

    #Combine all the experts - All scatter
    output_scatter = torch.zeros((B*T, d), device=device, dtype=input_dtype)
    for i in range(N):
        if expert_output_tensor[i] is not None:
            t_exp, _ = expert_output_tensor[i].shape
            for j in range(t_exp):
                output_scatter[expert_idx_tensor[i][j].item(),:] += expert_output_tensor[i][j,:]
    

    #Shared Experts
    for i in range(S):
        output_scatter += expert_ffn_fwd(input_tensor, f"layer0_ffn_sh_{i}")

    #Output Projection if required
    output_scatter = output_scatter.view(B,T,d)

    print("Done")

if __name__ == "__main__":
    main()