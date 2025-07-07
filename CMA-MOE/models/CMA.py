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
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Smaller gain
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
            nn.init.xavier_uniform_(module.weight, gain=0.1)
        
        for module in self.pos_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        
    def forward(self, feats, coords, mask, return_alpha=False):
        """
        return
          Z      : [L, D]  压缩 token
          alpha_list (可选) : list[(α, p)]，其中
                  α : [L, N] 稀疏注意力权重 (segment × patch)
                  p : [N, 2] 对应 patch 坐标，用于损失
        """
        # Handle edge case: all patches are masked
        if mask.all():
            return self.z.clone()
        
        feats = feats[~mask]     # [N, D]
        coords= coords[~mask]    # [N, 2]
        N, D  = feats.shape


        if N == 0:
            return self.z.clone()

        # Normalize input features to prevent overflow
        feats = F.layer_norm(feats, feats.shape[-1:])
        
        # Normalize coordinates to prevent large position bias
        coords_mean = coords.mean(dim=0, keepdim=True)
        coords_std = coords.std(dim=0, keepdim=True) + 1e-8  # Add epsilon to prevent division by zero
        coords = (coords - coords_mean) / coords_std

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

            z_l_acc = None
            attn_acc = []
            num_windows = 0
            
            for start in range(0, feats_seg.size(0), self.s):
                slc = slice(start, min(start+self.w, feats_seg.size(0)))
                f_win  = feats_seg[slc]                    # [w', D]
                c_win  = coords_seg[slc]                   # [w', 2]

                # Skip empty windows
                if f_win.size(0) == 0:
                    continue

                q = self.q(self.z[l:l+1])                  # [1, D]
                k = self.k(f_win)                          # [w', D]
                v = self.v(f_win)
                
                # Compute position bias more carefully
                c_mean = c_win.mean(0, keepdim=True)
                pos_bias = self.pos_mlp(c_win - c_mean)
                k = k + pos_bias

                # Stable attention computation with temperature scaling
                temperature = math.sqrt(self.z.size(1))
                attn_logits = (q @ k.T) / temperature
                
                # Clip logits to prevent overflow in softmax
                attn_logits = torch.clamp(attn_logits, min=-10, max=10)
                attn = F.softmax(attn_logits, dim=-1)

                z_win = attn @ v                             
                
                if z_l_acc is None:
                    z_l_acc = z_win.squeeze(0)
                else:
                    z_l_acc += z_win.squeeze(0)
                num_windows += 1
                
                if return_alpha:
                    full_alpha = torch.zeros(N, device=feats.device)
                    full_alpha[patch_global_idx[slc]] = attn.squeeze(0)
                    attn_acc.append(full_alpha)

            # Proper averaging
            if num_windows > 0:
                z_l_acc = z_l_acc / num_windows
            else:
                z_l_acc = self.z[l]
                
            z_out.append(z_l_acc)
            
            if return_alpha:
                if len(attn_acc) > 0:
                    seg_alpha = torch.stack(attn_acc).mean(0)
                else:
                    seg_alpha = torch.zeros(N, device=feats.device)
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
        nn.init.xavier_uniform_(self.memory_agg.weight, gain=0.1)
        nn.init.zeros_(self.memory_agg.bias)

    def forward(self, tokens_per_backbone):
        expert_outputs = []
        memory_updates = [torch.zeros_like(memory) for memory in self.memories]

        # Compute expert outputs and memory attentions
        for i, tokens in enumerate(tokens_per_backbone):
            # Handle edge case: empty tokens
            if tokens.numel() == 0:
                dummy_output = torch.zeros_like(tokens)
                expert_outputs.append(dummy_output)
                continue
                
            # Normalize tokens to prevent overflow
            tokens = F.layer_norm(tokens, tokens.shape[-1:])
            
            if self.num_experts > 1:
                context = torch.cat([tokens_per_backbone[j] for j in range(self.num_experts) if j != i], dim=0)
                
                if context.numel() > 0:
                    # Normalize context
                    context = F.layer_norm(context, context.shape[-1:])
                    
                    # Stable cross-attention computation
                    temperature = math.sqrt(tokens.size(-1))
                    cross_attn_logits = tokens @ context.T / temperature
                    cross_attn_logits = torch.clamp(cross_attn_logits, min=-10, max=10)
                    cross_attn_scores = F.softmax(cross_attn_logits, dim=-1)
                    cross_attn_output = cross_attn_scores @ context
                    
                    # Expert processing
                    expert_input = tokens + cross_attn_output
                else:
                    expert_input = tokens
            else:
                expert_input = tokens

            expert_output = self.experts[i](expert_input)

            memory_retrievals = []
            for l, memory in enumerate(self.memories):
                # Stable memory attention computation
                temperature = math.sqrt(self.token_dim)
                attn_logits = expert_output @ memory.T / temperature
                attn_logits = torch.clamp(attn_logits, min=-10, max=10)
                attn_scores = F.softmax(attn_logits, dim=-1)
                memory_retrieval = attn_scores @ memory
                memory_retrievals.append(memory_retrieval)

                # Update memory with stability check
                memory_update = attn_scores.T @ expert_output
                # Clip to prevent extreme values
                memory_update = torch.clamp(memory_update, min=-1.0, max=1.0)
                memory_updates[l] += memory_update

            # Aggregate multi-layer memory retrievals
            multi_memory_retrieval = self.memory_agg(torch.cat(memory_retrievals, dim=-1))

            expert_outputs.append(expert_output + multi_memory_retrieval)

        # Update memories with stable rule and clipping
        with torch.no_grad():
            for l in range(self.num_memory_layers):
                update = self.gamma * memory_updates[l]
                # Clip updates to prevent explosion
                update = torch.clamp(update, min=-0.1, max=0.1)
                self.memories[l].data = (1 - self.gamma) * self.memories[l].data + update

        # Compute gate weights with stability
        pooled_expert_outputs = []
        for e in expert_outputs:
            if e.numel() > 0:
                pooled = e.mean(dim=0)
            else:
                pooled = torch.zeros(self.token_dim, device=e.device)
            pooled_expert_outputs.append(pooled)
            
        gate_logits = torch.stack([po @ self.global_gate for po in pooled_expert_outputs], dim=0).squeeze(-1)
        # Clip gate logits to prevent overflow
        gate_logits = torch.clamp(gate_logits, min=-10, max=10)
        gate_weights = F.softmax(gate_logits, dim=0)

        # Weighted fusion
        fused_output = sum(w * e for w, e in zip(gate_weights, expert_outputs))

        return fused_output, gate_weights


class MultiModelFusionSystem(nn.Module):
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
        """
        Args:
            feats_list: list of [N_i, D_i] tensors from different backbone models
            coords_list: list of [N_i, 2] coordinate tensors (optional)
            pad_mask_list: list of [N_i] boolean masks (optional)
        Returns:
            logits: [B, n_classes] classification logits
        """
        compressed_tokens = []
        
        # Step 1 & 2: 维度对齐 + 序列压缩
        for i, feats in enumerate(feats_list):
            # Input validation
            if feats.numel() == 0:
                print(f"Warning: Empty feature tensor for model {i}")
                # Create dummy compressed tokens
                dummy_tokens = torch.zeros(self.num_tokens, self.token_dim, 
                                         device=feats.device, dtype=feats.dtype)
                compressed_tokens.append(dummy_tokens)
                continue
            
            # 维度对齐 - 使用预定义的对齐器
            aligned_feats = self.aligner(feats, backbone_idx=i)  # [N_i, D_i] -> [N_i, token_dim]
            
            # 序列压缩
            if coords_list is not None and pad_mask_list is not None and i < len(coords_list) and i < len(pad_mask_list):
                coords = coords_list[i]
                pad_mask = pad_mask_list[i]
                
                # Validate coordinates and masks
                if coords.numel() == 0 or pad_mask.numel() == 0:
                    print(f"Warning: Empty coords or mask for model {i}")
                    pooled = aligned_feats.mean(dim=0)
                    compressed = pooled.unsqueeze(0).repeat(self.num_tokens, 1)
                else:
                    try:
                        compressed = self.compressors[i](aligned_feats, coords, pad_mask)
                    except Exception as e:
                        print(f"Error in compressor {i}: {e}")
                        # Fallback to simple pooling
                        pooled = aligned_feats.mean(dim=0)
                        compressed = pooled.unsqueeze(0).repeat(self.num_tokens, 1)
            else:
                # 如果没有坐标信息，使用简单的平均池化并复制到num_tokens
                pooled = aligned_feats.mean(dim=0)  # [token_dim]
                compressed = pooled.unsqueeze(0).repeat(self.num_tokens, 1)  # [num_tokens, token_dim]

            
            compressed_tokens.append(compressed)
        
        # Step 3: 融合策略
        if self.fusion_type == "moe":
            fused_output, gate_weights = self.fusion(compressed_tokens)  # [num_tokens, token_dim]
        elif self.fusion_type == "concat":
            # 在最后一个维度concat: [num_experts, num_tokens, token_dim] -> [num_tokens, num_experts*token_dim]
            tokens_cat = torch.cat(compressed_tokens, dim=-1)  # [num_tokens, num_experts*token_dim]
            fused_output = self.fusion(tokens_cat)  # [num_tokens, token_dim]
        elif self.fusion_type == "self_attention":
            # 堆叠为 [num_tokens, num_experts, token_dim]
            tokens_stack = torch.stack(compressed_tokens, dim=1)  # [num_tokens, num_experts, token_dim]
            attn_out, _ = self.attn(tokens_stack, tokens_stack, tokens_stack)  # [num_tokens, num_experts, token_dim]
            attn_out = attn_out.reshape(attn_out.shape[0], -1)  # [num_tokens, num_experts*token_dim]
            fused_output = self.proj(attn_out)  # [num_tokens, token_dim]
        elif self.fusion_type == "cross_attention":
            # 以第一个模型为query，其余为key/value
            query = compressed_tokens[0].unsqueeze(1)  # [num_tokens, 1, token_dim]
            keys = torch.stack(compressed_tokens[1:], dim=1)  # [num_tokens, num_experts-1, token_dim]
            attn_out, _ = self.cross_attn(query, keys, keys)  # [num_tokens, 1, token_dim]
            fused_output = self.proj(attn_out.squeeze(1))  # [num_tokens, token_dim]
        
        # Step 4: MIL聚合 + 分类
        # 对所有tokens进行平均池化，然后分类
        pooled_features = fused_output.mean(dim=0)  # [token_dim]
        logits = self.mil_head(pooled_features)  # [n_classes]
        
        return logits.unsqueeze(0)  # 添加batch维度: [1, n_classes]


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
            feats: [N, D] feature tensor
            coords: [N, 2] coordinate tensor (unused for linear head)
            pad_mask: [N] boolean mask (unused for linear head)
        """
        # 维度对齐 - 使用预定义的对齐器
        aligned_feats = self.aligner(feats, backbone_idx=0)  # [N, D] -> [N, token_dim]
        
        # 聚合特征
        if self.head_type == "linear":
            # 简单平均池化
            pooled_features = aligned_feats.mean(dim=0)  # [token_dim]
        elif self.head_type == "mil":
            # MIL聚合 (这里简化为平均池化，可以扩展为注意力机制)
            pooled_features = aligned_feats.mean(dim=0)  # [token_dim]
        
        # 分类
        logits = self.head(pooled_features)  # [n_classes]
        
        return logits.unsqueeze(0)  # 添加batch维度: [1, n_classes]


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
            feats_list: list of [N_i, D_i] tensors from different backbone models
        """
        # 维度对齐 + 平均池化
        pooled_features = []
        for i, feats in enumerate(feats_list):
            aligned_feats = self.aligner(feats, backbone_idx=i)  # [N_i, D_i] -> [N_i, token_dim]
            pooled = aligned_feats.mean(dim=0)  # [token_dim]
            pooled_features.append(pooled)
        
        # 融合策略
        if self.fusion_type == "concat":
            # 直接concat
            fused_features = torch.cat(pooled_features, dim=0)  # [num_experts * token_dim]
            fused_output = self.fusion(fused_features)  # [token_dim]
        elif self.fusion_type == "self_attention":
            # 堆叠为 [1, num_experts, token_dim]
            features_stack = torch.stack(pooled_features, dim=0).unsqueeze(0)  # [1, num_experts, token_dim]
            attn_out, _ = self.attn(features_stack, features_stack, features_stack)  # [1, num_experts, token_dim]
            attn_out = attn_out.squeeze(0).reshape(-1)  # [num_experts * token_dim]
            fused_output = self.proj(attn_out)  # [token_dim]
        elif self.fusion_type == "cross_attention":
            # 以第一个模型为query，其余为key/value
            query = pooled_features[0].unsqueeze(0).unsqueeze(0)  # [1, 1, token_dim]
            keys = torch.stack(pooled_features[1:], dim=0).unsqueeze(0)  # [1, num_experts-1, token_dim]
            attn_out, _ = self.cross_attn(query, keys, keys)  # [1, 1, token_dim]
            fused_output = self.proj(attn_out.squeeze(0).squeeze(0))  # [token_dim]
        
        # 分类
        logits = self.head(fused_output)  # [n_classes]
        
        return logits.unsqueeze(0)  # 添加batch维度: [1, n_classes]
        return logits.unsqueeze(0)  # 添加batch维度: [1, n_classes]
