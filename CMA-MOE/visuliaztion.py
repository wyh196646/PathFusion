import torch
import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from finetune_utils import seed_torch, get_loader, slide_collate_fn
from datasets.slide_dataset import SlideDataset
from models.CMA import MultiModelFusionSystem

# 设置随机种子以保证复现性
seed_torch(torch.device("cuda" if torch.cuda.is_available() else "cpu"), seed=7)

# 加载基础模型
model_paths = ["model1.h5", "model2.h5"]  # 替换为你的模型路径
base_model_dims = [512, 512]  # 根据你的模型设置
num_experts = len(model_paths)

fusion_model = MultiModelFusionSystem(
    fusion_type='moe',
    num_experts=num_experts,
    token_dim=512,
    num_tokens=64,
    n_classes=2,
    base_model_feature_dims=base_model_dims
)

# 加载模型权重
for idx, path in enumerate(model_paths):
    with h5py.File(path, 'r') as hf:
        weights = {k: torch.tensor(v[:]) for k, v in hf.items()}
        fusion_model.load_state_dict(weights, strict=False)

fusion_model.eval()

# 加载WSI数据
csv_path = "dataset_csv/subtype/BRACS_COARSE.csv"  # 替换为你的csv路径
wsi_root = "/data4/embedding/BRACS"  # 替换为你的WSI根路径
dataset = pd.read_csv(csv_path)

# 示例选取单张slide进行可视化
slide_id = dataset['slide_id'].iloc[0]
slide_data = dataset[dataset['slide_id'] == slide_id]

# 实例化数据集
slide_dataset = SlideDataset(
    slide_data,
    wsi_root,
    [slide_id],
    task_cfg={"name": "classification", "label_dict": {"benign": 0, "malignant": 1}},
    split_key='slide_id',
    base_models=model_paths
)

loader = torch.utils.data.DataLoader(slide_dataset, batch_size=1, collate_fn=slide_collate_fn)

# 推理
with torch.no_grad():
    for batch in loader:
        imgs_list = [x for x in batch['imgs_list']]
        coords_list = [x for x in batch['coords_list']]
        pad_mask_list = [x for x in batch['pad_mask_list']]

        logits, _, _, _ = fusion_model(imgs_list, coords_list, pad_mask_list)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

# 提取坐标和概率用于绘制热图
coords = coords_list[0].cpu().numpy().squeeze()
pred_probs = probs.squeeze()[:, 1]  # malignant的概率

# 绘制热图
plt.figure(figsize=(10, 8))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=pred_probs, palette="Reds", edgecolor=None, alpha=0.7)
plt.title(f"WSI Prediction Heatmap for Slide ID: {slide_id}")
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # 根据WSI图像坐标特点
plt.colorbar(label='Malignancy Probability')
plt.grid(False)
plt.show()
