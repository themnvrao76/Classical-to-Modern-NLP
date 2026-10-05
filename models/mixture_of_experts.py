import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(num_experts))
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        router_logits = self.router(flat)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_values, dim=-1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(top_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat[token_idx])
            output.index_add_(0, token_idx, expert_out * weights[token_idx, slot_idx, None])
        probs = F.softmax(router_logits, dim=-1)
        importance = probs.mean(dim=0)
        load = F.one_hot(top_indices[:, 0], self.num_experts).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(importance * load)
        return output.view(original_shape), aux_loss


if __name__ == "__main__":
    moe = SparseMoE(d_model=256, d_ff=1024, num_experts=8, top_k=2)
    x = torch.randn(2, 32, 256)
    y, aux = moe(x)
    print("output:", tuple(y.shape), "aux_loss:", float(aux))
    print("parameters:", sum(p.numel() for p in moe.parameters()))
