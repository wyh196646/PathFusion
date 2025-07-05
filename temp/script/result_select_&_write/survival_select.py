import os
import seaborn as sns
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
import warnings
from utils import get_result_csv,select_best_models_with_shared_lr

eval_dir = '/home/yuhaowang/project/FMBC/downstream/finetune/outputs'
originl_result_save_dir = '../result/result_csv/original_learning_rate'
selected_result_save_dir = '../result/result_csv/survival_select'
if not os.path.exists(originl_result_save_dir):
    os.makedirs(originl_result_save_dir)
if not os.path.exists(selected_result_save_dir):
    os.makedirs(selected_result_save_dir)

def get_plot_results_optimized(eval_dir):
   
    evaluation_metrics = ['val_c_index']#, 
    csv = get_result_csv(eval_dir,evaluation_metrics)
    name = eval_dir.split('/')[-1]
    csv.to_csv(f'{originl_result_save_dir}/{name}.csv', index=False)
    final_df = select_best_models_with_shared_lr(csv, evaluation_metrics, evaluation_metrics, index=0,include_meanpool=True)
    
    final_df.to_csv(f'{selected_result_save_dir}/{name}.csv', index=False)
    
for task in os.listdir(eval_dir):
    # if task != 'TCGA-BRCA-SURVIVAL':
    #     continue
    try:
        get_plot_results_optimized(os.path.join(eval_dir, task))
    except:
        pass

