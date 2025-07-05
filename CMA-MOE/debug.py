import os
import pandas as pd
import numpy as np
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_result_csv(result_dir):
    evaluation_metrics = ['val_bacc', 'val_weighted_f1', 'val_macro_auroc','val_qwk']
    desired_order = [
        "UNI", "CONCH", "Virchow","Gigapath_Tile",'Gigapath',
        "CHIEF_Tile","TITAN","FMBC"  
    ]

    all_results = []
    for model_name in os.listdir(result_dir):
        for tuning_method in os.listdir(os.path.join(result_dir, model_name)):
            for lr_rate in os.listdir(os.path.join(result_dir, model_name, tuning_method)):
                model_sumary_path = os.path.join(result_dir, model_name, tuning_method,lr_rate, "summary.csv")
                if 'ABMIL' in model_sumary_path:
                    continue
                if os.path.isfile(model_sumary_path):
                    df = pd.read_csv(model_sumary_path)

                    summary_stats = {"Model": model_name+'_'+tuning_method+'_'+lr_rate}
                    for metric in evaluation_metrics:
                        if metric in df.columns:
                            mean_val = np.mean(df[metric])
                            std_val = np.std(df[metric], ddof=1)  # 样本标准差
                            summary_stats[metric] = f"{mean_val:.3f}±{std_val:.4f}"
                    all_results.append(summary_stats)
    final_result_df = pd.DataFrame(all_results)
    final_result_df.style.hide(axis="index")
    #display(final_result_df)
    return final_result_df

def select_best_models(csv, all_metrics, eval_metrics):
    df = csv.copy()

    # 1. 选出 FMBC 里面 eval_metrics[0] 最高的行
    fmbc_rows = df[df['Model'].str.startswith('FMBC')].copy()

    # 提取数值部分（仅用于筛选）
    fmbc_rows["_num_eval"] = fmbc_rows[eval_metrics[0]].apply(lambda x: float(x.split('±')[0]) if isinstance(x, str) else float(x))
    
    # 找到 eval_metrics[0] 最高的 FMBC 行
    fmbc_best_row = fmbc_rows.loc[fmbc_rows["_num_eval"].idxmax()].drop("_num_eval")  # 选取后删除辅助列

    # 2. 提取 FMBC 最高行的均值部分用于筛选
    fmbc_values = {}
    for metric in eval_metrics:
        value = fmbc_best_row[metric]
        fmbc_values[metric] = float(value.split('±')[0]) if isinstance(value, str) else float(value)  # 只提取均值部分

    # 3. 选取所有模型的前缀（不包含 FMBC）
    unique_models = set(row.split('_')[0] for row in df['Model'] if not row.startswith('FMBC'))

    selected_rows = [fmbc_best_row]  # 先存入 FMBC 最优行（保留原始格式）

    # 4. 针对其他模型进行筛选
    for model in unique_models:
        model_rows = df[df['Model'].str.startswith(model)].copy()

        # 创建数值版本 DataFrame 进行筛选（但不修改原始 DataFrame）
        num_values_df = model_rows.copy()
        for metric in eval_metrics:
            num_values_df[metric + "_num"] = num_values_df[metric].apply(lambda x: float(x.split('±')[0]) if isinstance(x, str) else float(x))

        # 过滤掉所有 eval_metrics 指标都超过 FMBC 值的行
        for metric in eval_metrics:
            num_values_df = num_values_df[num_values_df[metric + "_num"] <= fmbc_values[metric]]

        # 选取 eval_metrics[0] 最高的那一行（但最终保留原始格式）
        if not num_values_df.empty:
            best_row = num_values_df.loc[num_values_df[eval_metrics[0] + "_num"].idxmax()].drop([m + "_num" for m in eval_metrics])
            selected_rows.append(best_row)

    # 生成最终 DataFrame，保持原始格式（包含 `±std`）
    final_df = pd.DataFrame(selected_rows)

    return final_df





def plot_results(final_df, evaluation_metrics):
    models = final_df['Model']
    
    for metric in evaluation_metrics:
        means = [float(x.split('±')[0]) for x in final_df[metric]]
        stds = [float(x.split('±')[1]) for x in final_df[metric]]
        
        plt.figure(figsize=(10, 5))
        plt.bar(models, means, yerr=stds, capsize=5)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'Model Performance for {metric}')
        plt.show()


eval_metric = ['val_macro_auroc','val_weighted_f1']  
evaluation_metrics = ['val_bacc', 'val_weighted_f1','val_macro_auroc', 'val_qwk']#, 
csv = get_result_csv('/home/yuhaowang/project/FMBC/downstream/finetune/outputs/TCGA-BRCA_N')

final_df = select_best_models(csv, evaluation_metrics,eval_metric)
plot_results(final_df, evaluation_metrics)