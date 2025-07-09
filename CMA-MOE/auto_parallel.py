import subprocess
import time
import os

# ======== 配置参数 ========
gpus = [3,4,5,6,7]  # 你想使用的 GPU ID
max_jobs_per_gpu = 6  # 每个 GPU 同时运行的最大任务数

# ======== 所有命令（已填入） ========
commands1 = [
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "moe" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "concat" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "self_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "cross_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model UNI --input_dim 1024',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CHIEF --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Gigapath --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Virchow --input_dim 2560',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CONCH --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "moe" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "concat" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "self_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "cross_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
]
commands2 = [
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "moe" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "concat" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "self_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "cross_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model UNI --input_dim 1024',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CHIEF --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Gigapath --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Virchow --input_dim 2560',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CONCH --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "moe" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "concat" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "self_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv" --root_path "/data4/embedding/TCGA-BRCA" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "cross_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
]


commands3 = [
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "moe" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "concat" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "self_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "cross_attention" --base_models "CONCH" "UNI" "CHIEF_tile" --base_model_feature_dims 768 1024 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model UNI --input_dim 1024',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CHIEF --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Gigapath --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model Virchow --input_dim 2560',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model CONCH --input_dim 768',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "moe" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "concat" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "self_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
    'python main.py --task_cfg_path "task_configs/subtype/SLNBREAST_SUBTYPE.yaml" --dataset_csv "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv" --root_path "/data4/embedding/SLN-Breast" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output_2" --fusion_type "cross_attention" --base_models "Gigapath_tile" "Virchow" --base_model_feature_dims 1536 2560',
]

commands=commands1 + commands2 + commands3

# ======== 调度器实现 ========
gpu_job_count = {gpu: 0 for gpu in gpus}
gpu_processes = []

def find_available_gpu():
    # Find the GPU with the least number of running jobs
    available_gpus = [(gpu, count) for gpu, count in gpu_job_count.items() if count < max_jobs_per_gpu]
    if available_gpus:
        # Return the GPU with minimum job count
        return min(available_gpus, key=lambda x: x[1])[0]
    return None

def launch_command_on_gpu(command, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[LAUNCH] GPU {gpu} >> {command}")
    process = subprocess.Popen(command, shell=True, env=env)
    gpu_processes.append((gpu, process))
    gpu_job_count[gpu] += 1

def monitor_and_launch():
    idx = 0
    while idx < len(commands) or any(p.poll() is None for _, p in gpu_processes):
        # 清理完成的任务
        for i in range(len(gpu_processes) - 1, -1, -1):
            gpu, process = gpu_processes[i]
            if process.poll() is not None:
                print(f"[COMPLETE] GPU {gpu} finished a task. Current count: {gpu_job_count[gpu] - 1}")
                gpu_job_count[gpu] -= 1
                gpu_processes.pop(i)

        # 显示当前GPU状态
        if idx < len(commands):
            status = ", ".join([f"GPU{gpu}:{count}" for gpu, count in gpu_job_count.items()])
            print(f"[STATUS] {status} | Remaining tasks: {len(commands) - idx}")

        # 分配新任务
        while idx < len(commands):
            gpu = find_available_gpu()
            if gpu is not None:
                launch_command_on_gpu(commands[idx], gpu)
                idx += 1
            else:
                break
        time.sleep(5)

    print("[COMPLETE] All tasks finished!")

if __name__ == "__main__":
    monitor_and_launch()
