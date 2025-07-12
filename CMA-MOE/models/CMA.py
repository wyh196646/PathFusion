import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class MultiDimAligner(nn.Module):
    def __init__(self, input_dims, D_target=512):
        """
        Args:
            input_dims: list of input dimensions for each backbone model
            D_target: target dimension to align to
        """
        super().__init__()
        self.D_target = D_target
        self.input_dims = input_dims
        self.aligners = nn.ModuleList()
        
        # Pre-define aligners for each input dimension
        for D_in in input_dims:
            if D_in == D_target:
                aligner = nn.Identity()
            else:
                # Use more sophisticated MLP for better feature transformation
                aligner = nn.Sequential(
                    nn.Linear(D_in, D_target * 2, bias=True),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(D_target * 2, D_target, bias=True),
                    nn.LayerNorm(D_target)  # Add layer normalization for stability
                )
                # Initialize weights properly with smaller scale
                for layer in aligner:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.05)  # Smaller gain
                        nn.init.zeros_(layer.bias)
            self.aligners.append(aligner)

    def forward(self, x, backbone_idx):
        """
        Args:
            x: input tensor [N, D_in]
            backbone_idx: index of the backbone model (0, 1, 2, ...)
        Returns:
            aligned tensor [N, D_target]
        """
        if backbone_idx >= len(self.aligners):
            raise ValueError(f"backbone_idx {backbone_idx} out of range, only {len(self.aligners)} aligners available")
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected in input to aligner {backbone_idx}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        output = self.aligners[backbone_idx](x)
        
        # Check for NaN/Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: NaN/Inf detected in aligner {backbone_idx} output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output


class LSTCWA(nn.Module):
    def __init__(self, dim, num_tokens=64, win_size=64,
                 stride=32, heads=8):
        super().__init__()
        self.L  = num_tokens
        self.w  = win_size
        self.s  = stride
        # Initialize with smaller values to prevent overflow
        self.z  = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.q  = nn.Linear(dim, dim, bias=False)
        self.k  = nn.Linear(dim, dim, bias=False)
        self.v  = nn.Linear(dim, dim, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.heads = heads
        self.dim_h = dim // heads
        self.proj_out = nn.Linear(dim, dim)
        
        # Initialize weights with smaller scale
        for module in [self.q, self.k, self.v, self.proj_out]:
            nn.init.xavier_uniform_(module.weight, gain=0.05)
        
        for module in self.pos_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.05)
                nn.init.zeros_(module.bias)
        
    def forward(self, feats, coords, mask, return_alpha=False):
        B, N, D = feats.shape
        device = feats.device
        
        batch_outputs = []

        for b in range(B):
            sample_feats = feats[b]
            sample_mask = mask[b]
            
            valid_feats = sample_feats[~sample_mask]
            N_valid = valid_feats.shape[0]

            if N_valid == 0:
                sample_output = self.z.clone()
                batch_outputs.append(sample_output)
                continue
            
            valid_feats = F.layer_norm(valid_feats, valid_feats.shape[-1:])
            
            seg_size = max(1, N_valid // self.L)
            z_out = []
            for l in range(self.L):
                idx_start = l * seg_size
                idx_end = min((l + 1) * seg_size, N_valid)
                if idx_start >= N_valid:
                    z_out.append(self.z[l])
                    continue
                
                f_seg = valid_feats[idx_start:idx_end]
                q = self.q(self.z[l:l+1])
                k = self.k(f_seg)
                v = self.v(f_seg)

                temperature = math.sqrt(D)
                attn_logits = (q @ k.T) / temperature
                attn_logits = torch.clamp(attn_logits, -5, 5)
                attn = F.softmax(attn_logits, dim=-1)
                z_seg = attn @ v
                z_out.append(z_seg.squeeze(0))
            
            Z = torch.stack(z_out, 0)
            sample_output = self.proj_out(Z)
            batch_outputs.append(sample_output)
        
        return torch.stack(batch_outputs, 0)


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
                nn.TransformerEncoderLayer(d_model=token_dim, nhead=8, dropout=0.1), 
                num_layers=1
            )
            for _ in range(num_experts)
        ])

        # Hierarchical Memory (multiple layers) - Initialize with smaller values
        self.memories = nn.ParameterList([
            nn.Parameter(torch.randn(memory_size, token_dim) * 0.02)
            for _ in range(num_memory_layers)
        ])

        # Memory Aggregation layer
        self.memory_agg = nn.Linear(token_dim * num_memory_layers, token_dim)

        # Gating vector - Initialize with smaller values
        self.global_gate = nn.Parameter(torch.randn(token_dim, 1) * 0.02)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.memory_agg.weight, gain=0.05)
        nn.init.zeros_(self.memory_agg.bias)

    def forward(self, tokens_per_backbone):
        expert_outputs = []
        memory_updates = [torch.zeros_like(memory) for memory in self.memories]

        for i, tokens in enumerate(tokens_per_backbone):
            if tokens.numel() == 0:
                dummy_output = torch.zeros_like(tokens)
                expert_outputs.append(dummy_output)
                continue

            tokens = F.layer_norm(tokens, tokens.shape[-1:])

            if self.num_experts > 1:
                context = torch.cat([tokens_per_backbone[j] for j in range(self.num_experts) if j != i], dim=0)
                
                if context.numel() > 0:
                    context = F.layer_norm(context, context.shape[-1:])
                    
                    # 优化cross-attention为局部注意力，选最近的64个tokens
                    max_tokens = min(64, context.size(0))
                    context_subset = context[:max_tokens]
                    temperature = math.sqrt(tokens.size(-1))
                    cross_attn_logits = tokens @ context_subset.T / temperature
                    cross_attn_logits = torch.clamp(cross_attn_logits, -5, 5)
                    cross_attn_scores = F.softmax(cross_attn_logits, dim=-1)
                    cross_attn_output = cross_attn_scores @ context_subset
                    
                    expert_input = tokens + cross_attn_output
                else:
                    expert_input = tokens
            else:
                expert_input = tokens

            expert_output = self.experts[i](expert_input)

            memory_retrievals = []
            for l, memory in enumerate(self.memories):
                temperature = math.sqrt(self.token_dim)
                attn_logits = expert_output @ memory.T / temperature
                attn_logits = torch.clamp(attn_logits, -5, 5)
                attn_scores = F.softmax(attn_logits, dim=-1)
                memory_retrieval = attn_scores @ memory
                memory_retrievals.append(memory_retrieval)
                memory_updates[l] += attn_scores.T @ expert_output

            multi_memory_retrieval = self.memory_agg(torch.cat(memory_retrievals, dim=-1))
            expert_outputs.append(expert_output + multi_memory_retrieval)

        # memory update优化（每隔10步更新一次）
        if torch.rand(1).item() < 0.1:
            with torch.no_grad():
                for l in range(self.num_memory_layers):
                    update = self.gamma * memory_updates[l]
                    update = torch.clamp(update, -0.1, 0.1)
                    self.memories[l].data = (1 - self.gamma) * self.memories[l].data + update

        pooled_expert_outputs = [e.mean(dim=0) for e in expert_outputs]
        gate_logits = torch.stack([po @ self.global_gate for po in pooled_expert_outputs], dim=0).squeeze(-1)
        gate_logits = torch.clamp(gate_logits, -5, 5)
        gate_weights = F.softmax(gate_logits, dim=0)
        fused_output = sum(w * e for w, e in zip(gate_weights, expert_outputs))

        return fused_output, gate_weights



class  MultiModelFusionSystem(nn.Module):
    """
    完整的多模型融合系统:
    1. MultiDimAligner: 维度对齐
    2. LSTCWA: 序列压缩 (每个模型: [N, D] -> [64, 512])
    3. Fusion: 融合策略 (moe/concat/self_attention/cross_attention)
    4. MIL Head: 分类头
    """
    def __init__(self, 
                 fusion_type: str,
                 num_experts: int,
                 token_dim: int = 512,
                 num_tokens: int = 64,
                 n_classes: int = 2,
                 memory_size: int = 32,
                 num_memory_layers: int = 2,
                 gamma: float = 0.1,
                 mlp_hidden: int = 256,
                 attn_heads: int = 8,
                 win_size: int = 64,
                 stride: int = 32,
                 base_model_feature_dims: list = None):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.num_experts = num_experts
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        
        # Validate input dimensions
        if base_model_feature_dims is None:
            base_model_feature_dims = [token_dim] * num_experts
        if len(base_model_feature_dims) != num_experts:
            raise ValueError(f"base_model_feature_dims length {len(base_model_feature_dims)} must match num_experts {num_experts}")
        
        # 1. 维度对齐器 - 预先定义所有维度转换器
        self.aligner = MultiDimAligner(input_dims=base_model_feature_dims, D_target=token_dim)
        
        # 2. 序列压缩器 - 每个基础模型一个
        self.compressors = nn.ModuleList([
            LSTCWA(dim=token_dim, num_tokens=num_tokens, 
                   win_size=win_size, stride=stride)
            for _ in range(num_experts)
        ])
        
        # 3. 融合策略
        if self.fusion_type == "moe":
            self.fusion = HierarchicalMemoryMoE(num_experts, token_dim, memory_size, num_memory_layers, gamma)
        elif self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(num_experts * token_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
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
        
        # 4. MIL分类头 - Remove softmax for raw logits
        self.mil_head = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, n_classes)
            # Removed nn.Softmax - CrossEntropyLoss expects raw logits
        )
        
    def forward(self, feats_list, coords_list=None, pad_mask_list=None):
        B = feats_list[0].shape[0]
        batch_logits = []
        batch_expert_outputs = []
        batch_pooled_features = []
        batch_raw_pooled_features = []  # 新增，存储原始特征池化后的结果

        for b in range(B):
            sample_feats_list = [feats[b] for feats in feats_list]

            compressed_tokens = []
            raw_pooled = []  # 存储每个backbone原始特征池化后的向量

            for i, feats in enumerate(sample_feats_list):
                if feats.numel() == 0:
                    dummy_tokens = torch.zeros(self.num_tokens, self.token_dim, device=feats.device, dtype=feats.dtype)
                    compressed_tokens.append(dummy_tokens)
                    raw_pooled.append(torch.zeros(self.token_dim, device=feats.device))
                    continue

                aligned_feats = self.aligner(feats, backbone_idx=i)
                pooled_raw = aligned_feats.mean(dim=0)  # 原始特征池化（关键）
                raw_pooled.append(pooled_raw)
                compressed = pooled_raw.unsqueeze(0).repeat(self.num_tokens, 1)
                compressed_tokens.append(compressed)

            if self.fusion_type == "moe":
                fused_output, gate_weights = self.fusion(compressed_tokens)
            else:
                tokens_cat = torch.cat(compressed_tokens, dim=-1)
                fused_output = self.fusion(tokens_cat)

            pooled_features = fused_output.mean(dim=0)
            logits = self.mil_head(pooled_features)

            batch_logits.append(logits)
            batch_expert_outputs.append(fused_output)
            batch_pooled_features.append(pooled_features)
            batch_raw_pooled_features.append(torch.stack(raw_pooled).mean(dim=0))  # 原始特征池化再平均

        batch_logits = torch.stack(batch_logits, 0)
        batch_pooled_features = torch.stack(batch_pooled_features, 0)
        batch_raw_pooled_features = torch.stack(batch_raw_pooled_features, 0)

        return batch_logits, batch_expert_outputs, batch_pooled_features, batch_raw_pooled_features

class SingleModelBaseline(nn.Module):
    """
    单模型基线: 单个backbone + 线性头/MIL头
    """
    def __init__(self, 
                 token_dim: int = 512,
                 n_classes: int = 2,
                 head_type: str = "linear",
                 mlp_hidden: int = 256,
                 base_model_feature_dims: list = None):
        super().__init__()
        self.token_dim = token_dim
        self.head_type = head_type
        
        # 维度对齐 - 预先定义
        if base_model_feature_dims is None:
            base_model_feature_dims = [token_dim]
        self.aligner = MultiDimAligner(input_dims=base_model_feature_dims, D_target=token_dim)
        
        # 分类头
        if head_type == "linear":
            self.head = nn.Linear(token_dim, n_classes)
        elif head_type == "mil":
            self.head = nn.Sequential(
                nn.Linear(token_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, n_classes)
            )
        
    def forward(self, feats, coords=None, pad_mask=None):
        """
        Args:
            feats: [B, N, D] feature tensor batch
            coords: [B, N, 2] coordinate tensor batch (optional)
            pad_mask: [B, N] boolean mask batch (optional)
        Returns:
            logits: [B, n_classes] classification logits
        """
        B = feats.shape[0]
        batch_logits = []
        
        for b in range(B):
            # Extract single sample
            sample_feats = feats[b]  # [N, D]
            
            # 维度对齐
            aligned_feats = self.aligner(sample_feats, backbone_idx=0)
            
            # 聚合特征
            pooled_features = aligned_feats.mean(dim=0)  # [token_dim]
            
            # 分类
            logits = self.head(pooled_features)  # [n_classes]
            batch_logits.append(logits)
        
        batch_logits = torch.stack(batch_logits, 0)  # [B, n_classes]
        return batch_logits

class SimpleFusionBaseline(nn.Module):
    """
    简单融合基线: 多个backbone + 简单融合(concat/attention) + 分类头
    """
    def __init__(self, 
                 fusion_type: str,
                 num_experts: int,
                 token_dim: int = 512,
                 n_classes: int = 2,
                 mlp_hidden: int = 256,
                 attn_heads: int = 8,
                 base_model_feature_dims: list = None):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.num_experts = num_experts
        self.token_dim = token_dim
        
        # 维度对齐 - 预先定义
        if base_model_feature_dims is None:
            base_model_feature_dims = [token_dim] * num_experts
        self.aligner = MultiDimAligner(input_dims=base_model_feature_dims, D_target=token_dim)
        
        # 融合策略
        if self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(num_experts * token_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, token_dim)
            )
        elif self.fusion_type == "self_attention":
            self.attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
            self.proj = nn.Linear(token_dim, token_dim)
        elif self.fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
            self.proj = nn.Linear(token_dim, token_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # 分类头
        self.head = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, n_classes)
        )
        
    def forward(self, feats_list, coords_list=None, pad_mask_list=None):
        """
        Args:
            feats_list: list of [B, N_i, D_i] tensors from different backbone models
        Returns:
            logits: [B, n_classes] classification logits
        """
        B = feats_list[0].shape[0]
        batch_logits = []
        
        for b in range(B):
            # Extract single sample from batch
            sample_feats_list = [feats[b] for feats in feats_list]  # list of [N_i, D_i]
            
            # 维度对齐 + 平均池化
            pooled_features = []
            for i, feats in enumerate(sample_feats_list):
                if feats.numel() == 0:
                    # Handle empty features
                    pooled = torch.zeros(self.token_dim, device=feats.device, dtype=feats.dtype)
                else:
                    aligned_feats = self.aligner(feats, backbone_idx=i)  # [N_i, D_i] -> [N_i, token_dim]
                    pooled = aligned_feats.mean(dim=0)  # [token_dim]
                pooled_features.append(pooled)
            
            # 融合策略
            if self.fusion_type == "concat":
                # Concatenate all pooled features: [num_experts * token_dim]
                fused_features = torch.cat(pooled_features, dim=0)
                fused_output = self.fusion(fused_features)  # [token_dim]
            elif self.fusion_type == "self_attention":
                # Stack features for self-attention: [num_experts, token_dim]
                features_stack = torch.stack(pooled_features, dim=0).unsqueeze(0)  # [1, num_experts, token_dim]
                attn_out, _ = self.attn(features_stack, features_stack, features_stack)  # [1, num_experts, token_dim]
                attn_out = attn_out.squeeze(0)  # [num_experts, token_dim]
                fused_output = self.proj(attn_out.mean(dim=0))  # Average and project: [token_dim]
            elif self.fusion_type == "cross_attention":
                # Use first model as query, others as key/value
                query = pooled_features[0].unsqueeze(0).unsqueeze(0)  # [1, 1, token_dim]
                if len(pooled_features) > 1:
                    keys_values = torch.stack(pooled_features[1:], dim=0).unsqueeze(0)  # [1, num_experts-1, token_dim]
                    attn_out, _ = self.cross_attn(query, keys_values, keys_values)  # [1, 1, token_dim]
                    fused_output = self.proj(attn_out.squeeze(0).squeeze(0))  # [token_dim]
                else:
                    # Only one model, use it directly
                    fused_output = self.proj(pooled_features[0])  # [token_dim]
            
            # 分类
            logits = self.head(fused_output)  # [n_classes]
            batch_logits.append(logits)
        
        batch_logits = torch.stack(batch_logits, 0)  # [B, n_classes]
        return batch_logits

# 新增模块：Cross-Expert Memory Contrastive Loss (CEMCL)
class CrossExpertMemoryContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, expert_outputs):
        loss = 0
        M = len(expert_outputs)
        for m in range(M):
            pos = expert_outputs[m]
            neg = torch.cat([expert_outputs[n] for n in range(M) if n != m], dim=0)
            pos_sim = torch.exp((pos @ pos.T) / self.temperature)
            neg_sim = torch.exp((pos @ neg.T) / self.temperature).sum(dim=-1, keepdim=True)
            loss += -torch.log(pos_sim / (neg_sim + 1e-8)).mean()
        return loss / M

# 新增模块：Spatial-Aware Memory Routing (SAMR)
class SpatialAwareMemory(nn.Module):
    def __init__(self, dim, memory_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, dim))
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, features, coords):
        spatial_emb = self.pos_mlp(coords.mean(dim=0, keepdim=True))
        z_spatial = features + spatial_emb
        attn = F.softmax((z_spatial @ self.memory.T) / math.sqrt(features.size(-1)), dim=-1)
        return attn @ self.memory
