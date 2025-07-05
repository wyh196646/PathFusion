import os
import pandas as pd
import os
import seaborn as sns
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
import warnings

def get_result_csv(result_dir,evaluation_metrics):

    all_results = []
    for model_name in os.listdir(result_dir):#,
        # if  'tile' in model_name or model_name in ["TITAN","UNI","Virchow"] or 'UNI_Slide_25_cls' in model_name or 'Gigapath_tile_Slide' in model_name:# "CONCH","FMBC_Slide_100_cls","FMBC_Slide_100"
        #     continue #这是用来挑选分类任务的
        #     pass

        
        for tuning_method in os.listdir(os.path.join(result_dir, model_name)):
            # if tuning_method in ['LR_Different_MeanPool','LR_Same_MeanPool']:#,'LR_Same_Patch'
            #     continue 
            for lr_rate in os.listdir(os.path.join(result_dir, model_name, tuning_method)):
                if lr_rate == '0.1' and lr_rate == '0.01':
                    continue
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
    return final_result_df


def extract_numeric_values(df, eval_metrics):
    """为 DataFrame 添加数值列，去除 ± 后的字符串部分"""
    df = df.copy()
    for metric in eval_metrics:
        df[metric + '_num'] = df[metric].apply(lambda x: float(x.split('±')[0]) if isinstance(x, str) else float(x))
    return df



def find_val_predict_file(root_dir, model_name):
    """
    在根目录中查找匹配的 val_predict.csv 文件。
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if "prediction_results" in dirpath and "val_predict.csv" in filenames:
            # 解析路径，判断是否匹配所需模型
            path_parts = dirpath.split(os.sep)
            if model_name in path_parts:
                return os.path.join(dirpath, "val_predict.csv")
    return None

def extract_lr(model_name_str):
    """提取模型字符串中的学习率（最后一段）"""
    return model_name_str.split('_')[-1]


def group_fmbc_by_lr(df, eval_metrics, include_meanpool=False):
    """对 FMBC_Slide 和 FMBC_MeanPool 按学习率分组，并计算平均指标"""
    fmbc_df = df[df['Model'].str.contains('FMBC')]
    fmbc_df = extract_numeric_values(fmbc_df, eval_metrics)
    fmbc_df['lr'] = fmbc_df['Model'].apply(extract_lr)

    grouped_results = []

    for lr, group in fmbc_df.groupby('lr'):
        slide_group = group[group['Model'].str.startswith('FMBC_Slide')]
        meanpool_group = group[group['Model'].str.contains('MeanPool')]
        tile_group = group[group['Model'].str.contains('Same_Patch')]

        if include_meanpool:
            #if not slide_group.empty and not meanpool_group.empty and not tile_group.empty:
            if True:
                avg_metrics = {}
                for metric in eval_metrics:
                    avg = np.mean([
                        slide_group[metric + '_num'].mean(),
                        meanpool_group[metric + '_num'].mean(),
                        tile_group[metric + '_num'].mean()
                    ])
                    avg_metrics[metric] = avg

                grouped_results.append({
                    "lr": lr,
                    "slide_row": slide_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[0],
                    "meanpool_row": meanpool_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[0],
                    "tile_row": tile_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[0],   
                    "avg_metrics": avg_metrics
                })
        else:
            if not slide_group.empty:
                avg_metrics = {}
                for metric in eval_metrics:
                    avg_metrics[metric] = slide_group[metric + '_num'].mean()

                grouped_results.append({
                    "lr": lr,
                    "slide_row": slide_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[0],
                    "avg_metrics": avg_metrics
                })

    return grouped_results


def select_best_models_with_shared_lr(df, all_metrics, eval_metrics, index=0, include_meanpool=False):
    c = df.copy()

    grouped_fmbc = group_fmbc_by_lr(df, eval_metrics, include_meanpool)

    best_group = max(grouped_fmbc, key=lambda x: x["meanpool_row"][eval_metrics[0]])#avg_metrics
    bese_slide_group = max(grouped_fmbc, key=lambda x: x["slide_row"][eval_metrics[0]])#avg_metrics

    selected_lr = best_group["lr"]
    standby_lr = [grouped_fmbc[i]["lr"] for i in range(len(grouped_fmbc)) if grouped_fmbc[i]["lr"] != selected_lr]
    selected_rows = [bese_slide_group["slide_row"]]
    #selected_rows.append(best_group["tile_row"] 用来添加tile级别的mean pool用来下游任务的结果

    if include_meanpool and "meanpool_row" in best_group:
        selected_rows.append(best_group["meanpool_row"])

    non_fmbc_df = df[~df['Model'].str.contains('FMBC')]
    non_fmbc_df = extract_numeric_values(non_fmbc_df, eval_metrics)
    non_fmbc_df['lr'] = non_fmbc_df['Model'].apply(extract_lr)

   
    unique_models = set()
    for model in non_fmbc_df['Model']:
        if 'tile' not in model:
            unique_models.add(model.split('_')[0])
        else:
            unique_models.add(model.split('_')[0] + '_' + model.split('_')[1])

    for model in unique_models:
        model_group = non_fmbc_df[(non_fmbc_df['Model'].str.startswith(model)) & (non_fmbc_df['lr'] == selected_lr)]
        standby_group = non_fmbc_df[(non_fmbc_df['Model'].str.startswith(model)) & (non_fmbc_df['lr'].isin(standby_lr))]
        if not model_group.empty:
            best_row = model_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[index]
            standby_row = standby_group.sort_values(by=eval_metrics[0] + '_num', ascending=False).iloc[index]
            #select amx row to append
            if best_row[eval_metrics[0] + '_num'] > standby_row[eval_metrics[0] + '_num']:
                selected_rows.append(best_row)
            else:
                selected_rows.append(standby_row)
            
            #selected_rows.append(best_row)

    final_df = pd.DataFrame(selected_rows)
    return final_df
