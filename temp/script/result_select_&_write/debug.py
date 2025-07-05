#有几个基因计算spearman系数都是nan，我需要重新弄一下，重命名所有的csv结果，让我来重新弄一下
#主要就是CPTAC_GENE CPTAC_PROTEIN GTEX_BREST_GENE


import pandas as pd
import os
from pathlib import Path


def fill_average_metrics(file_path: str) -> None:
    """
    Fill missing values in 'val_average_spearman' and 'val_average_pearson'
    columns by computing the mean of the top 10% highest spearman correlation
    values in each row, and overwrite the CSV file in-place.

    Parameters:
    - file_path: Path to the CSV file to be processed.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    filled_df = df.copy()

    # Identify relevant spearman columns
    spearman_columns = [
        col for col in df.columns
        if col.startswith('val_spearman_') and not col.endswith('average_spearman')
    ]
    
    pearson_columns = [
        col for col in df.columns
        if col.startswith('val_pearson_') and not col.endswith('average_pearson')
    ]

    # Process each row individually
    for idx, row in df.iterrows():
        spearman_values = row[spearman_columns].dropna().sort_values(ascending=False)
        top_10_percent_count_spearman = max(1, int(len(spearman_values) * 0.10))
        mean_top_10_percent_spearman = spearman_values.head(top_10_percent_count_spearman).mean()
        
        pearson_values = row[pearson_columns].dropna().sort_values(ascending=False)
        top_10_percent_count_pearson = max(1, int(len(pearson_values) * 0.10))
        mean_top_10_percent_pearson = pearson_values.head(top_10_percent_count_pearson).mean()
        
        
        
        # Fill in missing average values if necessary
        if 'val_average_spearman' in df.columns :#and pd.isna(row['val_average_spearman'])
            filled_df.at[idx, 'val_average_spearman'] = mean_top_10_percent_spearman

        if 'val_average_pearson' in df.columns :#and pd.isna(row['val_average_pearson'])
            filled_df.at[idx, 'val_average_pearson'] = mean_top_10_percent_pearson

    # Overwrite the original CSV file
    filled_df.to_csv(file_path, index=False)


def find_summary_files(root_dir, verbose=False):
    """
    递归搜索所有层级中的 summary.csv 文件
    
    :param root_dir: 要搜索的根目录
    :param verbose: 是否显示搜索进度
    :return: 包含所有匹配文件路径的列表
    """
    matches = []
    root_path = Path(root_dir).expanduser().resolve()  # 处理 ~ 和相对路径
    
    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_path}")

    for i, file_path in enumerate(root_path.rglob("summary.csv")):
        try:
            if file_path.is_file():
                matches.append(str(file_path))
                if verbose and i % 100 == 0:  # 每100个文件打印一次进度
                    print(f"已扫描 {i+1} 个文件...")
        except PermissionError:
            if verbose:
                print(f"无权限访问: {file_path}")
    
    return matches

# 示例用法

target_dir = "/home/yuhaowang/project/FMBC/downstream/finetune/outputs" 
output_task= [ 'CPTAC_PROTEIN', 'CPTAC_GENE','GTEX_BREASTGENE']
processed_csv = []
for task in output_task:
    processed_csv.extend(find_summary_files(os.path.join(target_dir, task), verbose=True))

for file_path in processed_csv:
    try:
        fill_average_metrics(file_path)
        print(f"Processed {file_path} successfully.")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


