import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
from scipy import stats
warnings.filterwarnings('ignore')
import matplotlib.ticker as ticker 
colors = {
    "CONCH": "#A3DBD2",
    "CHIEF": "#99C0DB",
    "PRISM": "#CAC7E1",
    "Gigapath": "#DCDCDC",
    "FMBC_Slide": "#FDC281",
    "FMBC": "#FB998E",
}

selected_models = list(colors.keys())

def parser_name_to_summary(output_path, model_name):
    """与原先的逻辑保持一致，仅供示例使用。"""
    if 'FMBC' not in model_name:
        model, tuning, lr = model_name.split('_')
        csv_path = os.path.join(output_path, model, tuning, lr, "summary.csv")
    else:
        if 'Slide' in model_name:
            model, lr = model_name.split('_LR_')
            csv_path = os.path.join(output_path, model, 'LR', lr, "summary.csv")
        else:
            parts = model_name.split("_")
            model = parts[0]
            lr = parts[-1]
            csv_path = os.path.join(output_path, model, parts[1] + '_' + parts[2] + '_' + parts[3]+ '_' + parts[4], lr, "summary.csv")
    result = pd.read_csv(csv_path)
    return result

def plot_tasks_grid(
    tasks,
    evaluation_metric,
    compare_model='Gigapath',
    grid_shape=(2, 3),
    font_family='Arial',
    wspace=0.3,
    hspace=0.4,
    tight_layout=True,
    figure_size=(8, 8),
    csv_dir="/path/to/learning_rate_select",
    result_path="/path/to/finetune/outputs",
    save_dir='results',
    save_name='M_N_subplots.png',
    bar_width=0.2,          # 新增：柱子本身的宽度
    bar_gap=0.1,            # 新增：不同柱子之间的间距
    p_value_alternative='greater'  # 新增：Wilcoxon检验的假设方向
):
    """
    以 M×N 网格形式绘制多个子图，可进行配置：
    - bar_gap: 控制不同柱状条之间的距离
    - p_value_alternative: Wilcoxon检验是 'two-sided' 还是 'greater'/'less'
    """

    # 创建保存结果的文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 设置字体
    plt.rcParams['font.family'] = font_family

    # 创建 M×N 的子图网格
    M, N = grid_shape
    fig, axes = plt.subplots(M, N, figsize=figure_size)

    # 保证 axes 为二维数组 (即使只有1张图或1行/1列)
    if M == 1 and N == 1:
        axes = np.array([[axes]])
    elif M == 1 or N == 1:
        axes = axes.reshape(M, N)

    max_subplots = M * N
    tasks_to_plot = tasks[:max_subplots]
    def clean_model_name(model_name):
        model = model_name.split('_LR')[0]
        if 'FMBC' in model:
            if model == 'FMBC_Slide':
                return 'BRFoundf'  # Replace FMBC_Slide with BRFoundf
            else:
                return 'BRFound'  # Replace FMBC with BRFound
        else:
            return model
        
    for i, task in enumerate(tasks_to_plot):
        row = i // N
        col = i % N
        ax = axes[row, col]

        file_path = os.path.join(csv_dir, f"{task}.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] {file_path} not found, 跳过任务 {task}")
            continue

        df = pd.read_csv(file_path)

        # 清理模型名
        def clean_model_name(model_name):
            model = model_name.split('_LR')[0]
            if model == 'FMBC':
                return 'FMBC'
            elif 'FMBC' in model:
                return 'FMBC_Slide'
            else:
                return model
            
        def format_model_label(model):
            if model == "FMBC_Slide":
                return r"$\mathrm{BRFound}_V$"  # 使用 LaTeX 下标
            elif model == "FMBC":
                return r"$\mathrm{BRFound}$"
            else:
                return model
        def convert_evaluation_metric(metric):
            if metric == 'val_bacc':
                return 'BACC'
            if metric == 'val_macro_auroc':
                return 'MA-AUROC'

        df["Clean_Model"] = df["Model"].apply(clean_model_name)
        df.loc[df["Clean_Model"] == "FMBC_LR_Different_MeanPool", "Clean_Model"] = "FMBC_LR_Same_MeanPool"

        # 收集各模型在该任务上的平均值和 std
        values = []
        errors = []
        fold_results_each_model = []

        # 计算每个模型对应的折叠结果
        for model in selected_models:
            if model in df["Clean_Model"].values:
                row_data = df[df["Clean_Model"] == model].iloc[0]
                n_fold_result = parser_name_to_summary(os.path.join(result_path, task), row_data["Model"])
                fold_data = n_fold_result[evaluation_metric].values
                mean_val = np.mean(fold_data)
                std_val = np.std(fold_data)

                values.append(mean_val)
                errors.append(std_val)
                fold_results_each_model.append(fold_data)
            else:
                # 找不到该模型时，使用 0 占位
                values.append(0)
                errors.append(0)
                fold_results_each_model.append(np.array([0] * 5))

        # 生成 x 坐标位置
        x_positions = np.arange(len(selected_models)) * (bar_width + bar_gap)

        # 绘制柱状图、误差线和散点
        for idx, model in enumerate(selected_models):
            ax.bar(
                x_positions[idx],
                values[idx],
                width=bar_width,
                color=colors[model],
                edgecolor="none",
                alpha=0.85,
                label=model if i == 0 else None  # 仅在第一个子图显示图例
            )
            ax.errorbar(
                x_positions[idx],
                values[idx],
                yerr=errors[idx],
                ecolor="black",
                capsize=3,
                linewidth=1
            )
            # 添加散点
            jitter = np.random.uniform(-bar_width / 4, bar_width / 4, size=len(fold_results_each_model[idx]))
            ax.scatter(
                np.repeat(x_positions[idx], len(fold_results_each_model[idx])) + jitter,
                fold_results_each_model[idx],
                color="gray",
                alpha=0.6,
                s=12,
                zorder=10
            )

        # 在这里进行 Wilcoxon 检验（比较 best FMBC 与 compare_model
        # 主要使用了配对T检验
        fmbc_indices = [selected_models.index(m) for m in selected_models if m.startswith('FMBC')]
        if len(fmbc_indices) > 0 and compare_model in selected_models:
            compare_idx = selected_models.index(compare_model)
            best_fmbc_idx = None
            best_fmbc_val = float('-inf')

            # 选择最好的 FMBC 模型
            for fi in fmbc_indices:
                curr_mean = np.mean(fold_results_each_model[fi])
                if curr_mean > best_fmbc_val:
                    best_fmbc_val = curr_mean
                    best_fmbc_idx = fi

            if best_fmbc_idx is not None:
                # 计算两个模型的平均准确率之间的 P 值
                stat, p_value = stats.ttest_rel(
                    fold_results_each_model[best_fmbc_idx],
                    fold_results_each_model[compare_idx],
                )

                # 绘制 P 值括号
                max_error = max(errors[best_fmbc_idx], errors[compare_idx])
                y_max = max(values[best_fmbc_idx], values[compare_idx]) + max_error + 0.05
                ax.plot(
                    [x_positions[best_fmbc_idx], x_positions[best_fmbc_idx],
                    x_positions[compare_idx], x_positions[compare_idx]],
                    [y_max, y_max + 0.01, y_max + 0.01, y_max],
                    lw=1, color='black'
                )
                ax.text(
                    (x_positions[best_fmbc_idx] + x_positions[compare_idx]) / 2,
                    y_max + 0.015,
                    f"p={p_value:.3g}",
                    ha='center',
                    fontsize=8
                )

        # 设置 x 轴刻度
        ax.set_xticks(x_positions)
        ax.set_xticklabels([format_model_label(m) for m in selected_models], rotation=45, ha="right", fontsize=10)

        # 设置子图标题
        ax.set_title(f"{task}", fontsize=12, pad=10,fontweight='bold')

        # y 轴标签
        ax.set_ylabel(convert_evaluation_metric(evaluation_metric))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        # 隐藏上、右边框
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 拿第一个子图的 handles & labels 作为图例
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # 布局设置
    if tight_layout:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # 保存图像并显示
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.show()
    plt.close(fig)
