import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class MultiDimAligner(nn.Module):

    def __init__(self, D_target=512):
        super().__init__()
        self.D_target = D_target
        self.aligners = nn.ModuleDict()   # key = str(D_in)

    def forward(self, x):
        D_in = x.shape[-1]
        key = str(D_in)
        if key not in self.aligners:
            if D_in == self.D_target:
                self.aligners[key] = nn.Identity()
            else:
                lin = nn.Linear(D_in, self.D_target, bias=False)
                nn.init.xavier_uniform_(lin.weight)
                self.aligners[key] = lin
            # 新增：将新模块转移到当前设备
            self.aligners[key] = self.aligners[key].to(x.device)  # 关键修复
        return self.aligners[key](x)


class LSTCWA(nn.Module):
    def __init__(self, dim, num_tokens=64, win_size=64,
                 stride=32, heads=8):
        super().__init__()
        self.L  = num_tokens
        self.w  = win_size
        self.s  = stride
        self.z  = nn.Parameter(torch.randn(num_tokens, dim))
        self.q  = nn.Linear(dim, dim, bias=False)
        self.k  = nn.Linear(dim, dim, bias=False)
        self.v  = nn.Linear(dim, dim, bias=False)
        self.pos_mlp = nn.Sequential(nn.Linear(2, dim),
                                     nn.ReLU(),
                                     nn.Linear(dim, dim))
        self.heads = heads
        self.dim_h = dim // heads
        self.proj_out = nn.Linear(dim, dim)
        
    def forward(self, feats, coords, mask, return_alpha=False):
        """
        return
          Z      : [L, D]  压缩 token
          alpha_list (可选) : list[(α, p)]，其中
                  α : [L, N] 稀疏注意力权重 (segment × patch)
                  p : [N, 2] 对应 patch 坐标，用于损失
        """
        feats = feats[~mask]     # [N, D]
        coords= coords[~mask]    # [N, 2]
        N, D  = feats.shape

        seg_id = torch.div(torch.arange(N, device=feats.device)*self.L,
                           N, rounding_mode='floor')   # [N]
        z_out, alpha_collect = [], []

        for l in range(self.L):
            idx = (seg_id == l)
            if idx.sum() == 0:
                z_out.append(self.z[l])
                if return_alpha:
                    alpha_collect.append((torch.zeros(N, device=feats.device),
                                           coords))   # 空 α
                continue

            feats_seg  = feats[idx]
            coords_seg = coords[idx]           # [T_seg, 2]
            # 记录 segment → 原 patch 映射索引
            patch_global_idx = torch.nonzero(idx).squeeze(1)  # [T_seg]

            z_l_acc, attn_acc = 0., []
            for start in range(0, feats_seg.size(0), self.s):
                slc = slice(start, min(start+self.w, feats_seg.size(0)))
                f_win  = feats_seg[slc]                    # [w', D]
                c_win  = coords_seg[slc]                   # [w', 2]

              
                q = self.q(self.z[l:l+1])                  # [1, D]
                k = self.k(f_win)                          # [w', D]
                v = self.v(f_win)
                pos_bias = self.pos_mlp(c_win - c_win.mean(0, keepdim=True))
                k = k + pos_bias

                attn = (q @ k.T) / math.sqrt(self.z.size(1)) 
                attn = attn.softmax(-1)                       

                z_win = attn @ v                             
                z_l_acc += z_win.squeeze(0)
                if return_alpha:
                    full_alpha = torch.zeros(N, device=feats.device)
                    full_alpha[patch_global_idx[slc]] = attn.squeeze(0)
                    attn_acc.append(full_alpha)

            z_out.append(z_l_acc / max(1, len(attn_acc)))
            if return_alpha:
                
                seg_alpha = torch.stack(attn_acc).mean(0)  
                alpha_collect.append((seg_alpha, coords)) 

        Z = torch.stack(z_out, 0)  # [L, D]

        if return_alpha:
            # 拼成 [L, N]
            alpha_map = torch.stack([a for a,_ in alpha_collect], 0)
            return self.proj_out(Z), (alpha_map, coords) 
        else:
            return self.proj_out(Z)



class HierarchicalMemoryMoE(nn.Module):
    def __init__(self, num_experts, token_dim, memory_size, num_memory_layers, gamma=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.token_dim = token_dim
        self.memory_size = memory_size
        self.num_memory_layers = num_memory_layers
        self.gamma = gamma

        # Expert Networks
        self.experts = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=token_dim, nhead=8), 
                num_layers=1
            )
            for _ in range(num_experts)
        ])

        # Hierarchical Memory (multiple layers)
        self.memories = nn.ParameterList([
            nn.Parameter(torch.randn(memory_size, token_dim))
            for _ in range(num_memory_layers)
        ])

        # Memory Aggregation layer
        self.memory_agg = nn.Linear(token_dim * num_memory_layers, token_dim)

        # Gating vector
        self.global_gate = nn.Parameter(torch.randn(token_dim, 1))

    def forward(self, tokens_per_backbone):
        expert_outputs = []
        memory_updates = [torch.zeros_like(memory) for memory in self.memories]

        # Compute expert outputs and memory attentions
        for i, tokens in enumerate(tokens_per_backbone):
            context = torch.cat([tokens_per_backbone[j] for j in range(self.num_experts) if j != i], dim=0)

            # Cross-attention between backbones
            cross_attn_scores = F.softmax(tokens @ context.T / tokens.size(-1)**0.5, dim=-1)
            cross_attn_output = cross_attn_scores @ context

            # Expert processing
            expert_input = tokens + cross_attn_output
            expert_output = self.experts[i](expert_input)

            memory_retrievals = []
            for l, memory in enumerate(self.memories):
                # Memory attention
                attn_scores = F.softmax(expert_output @ memory.T / self.token_dim**0.5, dim=-1)
                memory_retrieval = attn_scores @ memory
                memory_retrievals.append(memory_retrieval)

                # Update memory
                memory_updates[l] += attn_scores.T @ expert_output

            # Aggregate multi-layer memory retrievals
            multi_memory_retrieval = self.memory_agg(torch.cat(memory_retrievals, dim=-1))

            expert_outputs.append(expert_output + multi_memory_retrieval)

        # Update memories with stable rule
        with torch.no_grad():
            for l in range(self.num_memory_layers):
                self.memories[l].data = (1 - self.gamma) * self.memories[l].data + self.gamma * memory_updates[l]

        # Compute gate weights
        pooled_expert_outputs = [e.mean(dim=0) for e in expert_outputs]
        gate_logits = torch.stack([po @ self.global_gate for po in pooled_expert_outputs], dim=0).squeeze(-1)
        gate_weights = F.softmax(gate_logits, dim=0)

        # Weighted fusion
        fused_output = sum(w * e for w, e in zip(gate_weights, expert_outputs))

        return fused_output, gate_weights


class FusionModel(nn.Module):
    """
    支持多种融合方式的多模型融合类
    fusion_type: 'moe' | 'concat' | 'self_attention' | 'cross_attention'
    """
    def __init__(self, 
                 fusion_type: str,
                 num_experts: int,
                 token_dim: int,
                 memory_size: int = 32,
                 num_memory_layers: int = 2,
                 gamma: float = 0.1,
                 mlp_hidden: int = 256,
                 attn_heads: int = 8):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.num_experts = num_experts
        self.token_dim = token_dim

        if self.fusion_type == "moe":
            self.fusion = HierarchicalMemoryMoE(num_experts, token_dim, memory_size, num_memory_layers, gamma)
        elif self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(num_experts * token_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, token_dim)
            )
        elif self.fusion_type == "self_attention":
            self.attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
            self.proj = nn.Linear(num_experts * token_dim, token_dim)
        elif self.fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
            self.proj = nn.Linear(token_dim, token_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, tokens_per_backbone):
        """
        tokens_per_backbone: list of [N, D] tensors, N为token数，D为token_dim
        返回: fused_output, (可选: gate_weights)
        """
        if self.fusion_type == "moe":
            return self.fusion(tokens_per_backbone)
        elif self.fusion_type == "concat":
            # [num_experts, N, D] -> [N, num_experts*D]
            tokens_cat = torch.cat(tokens_per_backbone, dim=-1)
            fused = self.fusion(tokens_cat)
            return fused, None
        elif self.fusion_type == "self_attention":
            # 堆叠为 [N, num_experts, D]
            tokens_stack = torch.stack(tokens_per_backbone, dim=1)
            # 变成 [N, num_experts, D] -> [N, num_experts, D]
            attn_out, _ = self.attn(tokens_stack, tokens_stack, tokens_stack)
            attn_out = attn_out.reshape(attn_out.shape[0], -1)
            fused = self.proj(attn_out)
            return fused, None
        elif self.fusion_type == "cross_attention":
            # 以第一个模型为query，其余为key/value
            query = tokens_per_backbone[0].unsqueeze(1)  # [N, 1, D]
            keys = torch.stack(tokens_per_backbone[1:], dim=1)  # [N, num_experts-1, D]
            attn_out, _ = self.cross_attn(query, keys, keys)
            fused = self.proj(attn_out.squeeze(1))
            return fused, None
        else:
            raise NotImplementedError
