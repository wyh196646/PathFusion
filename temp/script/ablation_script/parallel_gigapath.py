import os
import subprocess
import time
tasks = {
    "TCGA-BRCA-SUBTYPE": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml"
    },
    "CAMELYON16_TEST_CANCER": {
        "embedding_dir": "/data4/embedding/CAMELYON16",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON16_TEST_CANCER.yaml"
    },
    "DORID_2": {
        "embedding_dir": "/data4/embedding/DORID",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/DORID_2.yaml"
    },
    "TCGA-BRCA_T": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_T.yaml"
    },
    "BRACS_COARSE": {
        "embedding_dir": "/data4/embedding/BRACS",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BRACS_COARSE.yaml"
    },
    "AHSL-NON-IDC-GRADE": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-NON-IDC-GRADE.yaml"
    },
    "CAMELYON16_TEST_IDC": {
        "embedding_dir": "/data4/embedding/CAMELYON16",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON16_TEST_IDC.yaml"
    },
    "BCNB_TUMOR": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BCNB_TUMOR.yaml"
    },
    "BCNB_ALN": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BCNB_ALN.yaml"
    },
    "AIDPATH_IDC": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AIDPATH_IDC.yaml"
    },
    "CAMELYON17_STAGE_4SUBTYPING": {
        "embedding_dir": "/data4/embedding/CAMELYON17",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON17_STAGE_4SUBTYPING.yaml"
    },
    "IMPRESS_RESIDUAL-TUMOR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/IMPRESS_RESIDUAL-TUMOR.yaml"
    },
    "AHSL-GRADE-3": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-GRADE-3.yaml"
    },
    "BRACS_FINE": {
        "embedding_dir": "/data4/embedding/BRACS",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BRACS_FINE.yaml"
    },
    "AHSL-GRADE-1": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-GRADE-1.yaml",
        "folds": 1,
    },
    "SLNBREAST_SUBTYPE": {
        "embedding_dir": "/data4/embedding/SLN-Breast",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/SLNBREAST_SUBTYPE.yaml"
    },
    "TCGA-BRCA_STAGE": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_STAGE.yaml"
    },
    "POST-NAT-BRCA-3TYPE": {
        "embedding_dir": "/data4/embedding/Post-NAT-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/POST-NAT-BRCA-3TYPE.yaml",
        "folds":1
    },
    "TCGA-BRCA_M": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_M.yaml"
    },
    "DORID_6": {
        "embedding_dir": "/data4/embedding/DORID",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/DORID_6.yaml"
    },
    "CPTAC_IDC": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CPTAC_IDC.yaml"
    },
    "TUPAC_TUMOR_SCORE": {
        "embedding_dir": "/data4/embedding/TUPAC",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TUPAC_TUMOR_SCORE.yaml"
    },
    "AIDPATH_GRADE": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AIDPATH_GRADE.yaml"
    },
    "TCGA-BRCA_N": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_N.yaml"
    },
    "POST-NAT-BRCA-HERIHC": {
        "embedding_dir": "/data4/embedding/Post-NAT-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/POST-NAT-BRCA-HERIHC.yaml",
        "folds":1
    },
    "CAMELYON17_STAGE": {
        "embedding_dir": "/data4/embedding/CAMELYON17",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON17_STAGE.yaml"
    },
    "TCGA-BRCA_MOLECULAR_SUBTYPING": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_MOLECULAR_SUBTYPING.yaml"
    },
    "IMPRESS_PCR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PCR.yaml"
    },
    "IMPRESS_PD-L1-TUMOR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PD-L1-TUMOR.yaml"
    },
    "IMPRESS_CD8-PERITUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD8-PERITUMORAL.yaml"
    },
    "IMPRESS_PD-L1-STROMA": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PD-L1-STROMA.yaml"
    },
    "IMPRESS_PR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PR.yaml"
    },
    "IMPRESS_CD163-PERITUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD163-PERITUMORAL.yaml"
    },
    "BCNB_HER2": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_HER2.yaml"
    },
    "AIDPATH_KI67PRED": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_KI67PRED.yaml"
    },
    "POST-NAT-BRCA-PR": {
        "embedding_dir": "/data4/embedding/Post-NAT-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-BRCA-PR.yaml",
        "folds":1
    },
    "BCNB_ER": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_ER.yaml"
    },
    "AIDPATH_CERB2": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_CERB2.yaml"
    },
    "IMPRESS_CD163-INTRATUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD163-INTRATUMORAL.yaml"
    },
    "BCNB_PR": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_PR.yaml"
    },
    "POST-NAT-BRCA-ER": {
        "embedding_dir": "/data4/embedding/Post-NAT-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-BRCA-ER.yaml",
        "folds":1
    },
    "IMPRESS_CD8-INTRATUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD8-INTRATUMORAL.yaml"
    },
    "AIDPATH_RESTR": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_RESTR.yaml"
    },
    "POST-NAT-BRCA-ANTIHER2": {
        "embedding_dir": "/data4/embedding/Post-NAT-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-BRCA-ANTIHER2.yaml",
        "folds":1
    },
    "CPTAC_AJCC8SUBTYPE": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CPTAC_AJCC8SUBTYPE.yaml",
        "folds": 1
    },
    "IMPRESS_ER": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_ER.yaml"
    },
    "TCGA-BRCA_TP53": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_TP53.yaml"
    },
    "TCGA-BRCA_HRD": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_HRD.yaml"
    },
    "TCGA-BRCA_CDH": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_CDH.yaml"
    },
    "TCGA-BRCA_GATA-3": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_GATA-3.yaml"
    },
    "TCGA-BRCA_MAP3K1": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_MAP3K1.yaml"
    },
    "TCGA-BRCA_TTN": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_TTN.yaml"
    },
    "TCGA-BRCA_MUC16": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_MUC16.yaml"
    },
    "TCGA-BRCA_MUC16": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_MUC16.yaml"
    },
    "TCGA-BRCA_CDH": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_CDH.yaml"
    },
    'MULTI_OMICS_ASCAT-PURITY':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_ASCAT-PURITY.yaml"
    },
    'MULTI_OMICS_FRACTION_CANCER':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_FRACTION_CANCER.yaml"
    },
    'MULTI_OMICS_ASCAT-PURITY':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_ASCAT-PURITY.yaml"
    },
    'MULTI_OMICS_HRD':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_HRD.yaml"
    },
    'MULTI_OMICS_IPS':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_IPS.yaml"
    },
    'MULTI_OMICS':{
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS.yaml"
    },
    'TCGA-BRCA_GENE-EXP':{
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/TCGA-BRCA_GENE-EXP.yaml"
    },
    
    "CPTAC_PROTEIN": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/CPTAC_PROTEIN.yaml"
    },
    "TCGA-BRCA_PROTEIN": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/TCGA-BRCA_PROTEIN.yaml"
    },
    "MULTI_OMICS_HRD": {
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_HRD.yaml"
    },
    "TUPAC_MOLECULAR_SCORE": {
        "embedding_dir": "/data4/embedding/TUPAC",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/TUPAC_MOLECULAR_SCORE.yaml"
    },
    "GTEX_BREASTGENE": {
        "embedding_dir": "/data4/embedding/GTEX_Breast",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/GTEX_BREASTGENE.yaml"
    },
    "MULTI_OMICS_ASCAT-PURITY": {
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_ASCAT-PURITY.yaml"
    },
    "CPTAC_GENE": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/CPTAC_GENE.yaml"
    },
    "TCGA-BRCA_GENE-EXP": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/TCGA-BRCA_GENE-EXP.yaml"
    },
    "MULTI_OMICS": {
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS.yaml"
    },
    "MULTI_OMICS_FRACTION_CANCER": {
        "embedding_dir": "/data4/embedding/Multi-omic",
        "csv_dir": "dataset_csv/gene_exp",
        "task_cfg": "task_configs/gene_exp/MULTI_OMICS_FRACTION_CANCER.yaml"
    },
    "AHSL_SUBTYPE": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL_SUBTYPE.yaml"
    },
        "TCGA-BRCA_PIK3CA": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_PIK3CA.yaml"
    },

    "GTEX_PROG": {
        "embedding_dir": "/data4/embedding/GTEX_Breast",
        "csv_dir": "dataset_csv/survival",
        "task_cfg": "task_configs/survival/GTEX_PROG.yaml",
    },
    "TCGA-BRCA-SURVIVAL": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/survival",
        "task_cfg": "task_configs/survival/TCGA-BRCA-SURVIVAL.yaml",
        "batch_size": 10
    }
}

gpu_config = {
    0: 3,
    1: 3, 
    2: 3,  
    3: 3,  
    4: 3, 
    5: 3, 
    6: 3,
    7: 3,
}
pretrain_models = ['FMBC']
pretrain_model_dim_dict = {

    "FMBC":1024,
}
pretrain_model_types_dict = {

    "FMBC": "slide_level"
}
def get_tuning_methods(pretrain_model):

    # 根据预训练模型的不同，返回不同的调参方法
    if pretrain_model =='FMBC':
        # 定义调参方法的组合
        combinations = [
            f"LR_{lr}_{pool}"
            for lr in [ 'Same_0.25']#"Frozen","Same", "Different",,'Same_1'
            for pool in ["MeanPool" ]#"CLSPool"
        ]
    # combinations = []
    # combinations.append("LR_Same_Patch")
    return combinations


def get_available_gpus():
    return list(gpu_config.keys())

def is_task_hanging(process):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], 
                            capture_output=True, text=True)
    output = result.stdout.strip()
    if str(process.pid) in output and "0 MiB" in output:
        return True
    return False

def run_task(task_name, command, gpu_id):
    print(f"Starting: {command} on GPU {gpu_id}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, shell=True, env=env)
    running_tasks[gpu_id].append((process, command))

gpu_list = get_available_gpus()
running_tasks = {gpu: [] for gpu in gpu_list}

import yaml
tasks = yaml.load(open('all_task.yaml', 'r'), Loader=yaml.FullLoader)
task_queue = []
slide_weight_path= '/home/yuhaowang/project/FMBC/SlideModel/slide_gigapath/checkpoint0090.pth'
ablation_model = 'Gigapath'
learning_rates = [0.001, 0.0001]
for task_name, config in tasks.items():#
    embedding_dir = config["embedding_dir"]
    csv_dir = config["csv_dir"]
    task_cfg = config["task_cfg"]
    if "batch_size" in config.keys():
        batch_size = config['batch_size']
    else:
        batch_size = 1
    for pretrain_model in pretrain_models:
        pretrain_model_type = pretrain_model_types_dict[pretrain_model]
        tuning_methods = get_tuning_methods(pretrain_model)
        input_dim = pretrain_model_dim_dict[pretrain_model]
        root_path = os.path.join(embedding_dir, pretrain_model)
        dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")
        
        for tuning_method in tuning_methods:
            for learning_rate in learning_rates:
               
                   
                output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, str(learning_rate), 'summary.csv')
                if ablation_model == 'UNI':
                    output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, str(learning_rate), 'summary.csv').replace('FMBC', 'UNI')
                if ablation_model == 'Gigapath':
                    output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, str(learning_rate), 'summary.csv').replace('FMBC', 'Gigapath_tile')
                if os.path.exists(output_prediction):
                    print(f"Skipping task: {output_prediction} already exists")
                    continue
                
                command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                        f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                        f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method} --lr {learning_rate} --pretrained {slide_weight_path}\
                            --batch_size {batch_size} --ablation_model {ablation_model}"
                task_queue.append((task_name, command))

while task_queue or any(len(v) > 0 for v in running_tasks.values()):
    for gpu in gpu_list:
        running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
        if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
            task_name, cmd = task_queue.pop(0)
            run_task(task_name, cmd, gpu)
    
    time.sleep(10)

print("All tasks completed.")


