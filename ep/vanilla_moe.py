import torch
import torch.nn as nn
import torch.nn.functional as F

def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up

class ExpertMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(swiglu(self.gate_up_proj(x)))
        
class MOELayer(nn.Module):
    def __init__(self, k, d_model=4096, d_ffn=16384, num_expert=1):
        super().__init__()
        self.num_expert = num_expert
        self.k = k
        # 3 * d_model * d_ff == 8 * d_model ** 2 ==> d_ff = 8/3 * d_model
        # vanilla moe contains router (d_model, num_experts), SwiGLU(num_expert, d_model, 2 * d_moe_ffn), down(num_expert, d_moe_ffn, d_model) 
        # dist env: SwiGLU(num_expert / ep, d_model, 2 * d_ff / num_expert), down(num_expert / ep, d_ff / num_expert, d_model)
        self.router = nn.Linear(d_model, num_expert)
        assert(d_ffn // num_expert == d_ffn / num_expert, f"{d_ffn=} must be divisible by {num_expert=}")
        self.experts = nn.ModuleList(ExpertMLP(d_model, d_ffn // num_expert) for _ in range(num_expert))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] => [T, D]
        B, S, D = x.shape
        x = x.reshape(B * S, D)
        # router: linear [T, D] => [T, E], topk, softmax
        expert_logits = self.router(x)
        expert_logits, selected_experts = torch.topk(expert_logits, self.k)
        expert_probs = F.softmax(expert_logits, dim=-1)
        # expert_logits, selected_experts: [T, k]

        # expert dispatch:
        # input: x:(tokens) [T, D], expert_probs(token_topk_probs) [T, k], selected_experts(token_topk_idx) [T, k]
        # output: y (tokens) [B, S, D] <- [T, D]
        y = torch.new_zeros(B * S, D)

        for i in range(self.k):
            expert_weights = expert_probs[:, i] # [T]
            expert_indecies = selected_experts[:, i] # [T]

            # expert calculation and combine
            for j in expert_indecies.unique():
                mask = j == expert_indecies # [T]
                expert_out = self.experts[j](x[mask]) # Q: expert_indecies has shape [T] => select j, expert_out [n, D]
                expert_out *= expert_weights[mask].unsqueeze(-1)
                y[mask] += expert_out

        return y.reshape(B, S, D)
