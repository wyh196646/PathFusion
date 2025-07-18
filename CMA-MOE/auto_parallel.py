import subprocess
import time
import os

# ======== 配置参数 ========
gpus = [0,1,2,3,4,5,6,7]  # 你想使用的 GPU ID
max_jobs_per_gpu = 4  # 每个 GPU 同时运行的最大任务数

# ======== 所有命令（已填入） ========

def generate_commands(task_configs):
    commands = []
    
    # Common parameters
    common_params = {
        'pretrain_model_type': 'slide_level',
        'tuning_method': 'MIL',
        'batch_size': '1',
    }
    
    # Multi-model fusion configurations
    fusion_configs = [
        {'fusion_type': 'moe', 'lr': '0.0001'},
        # {'fusion_type': 'concat', 'lr': '0.0001'},
        # {'fusion_type': 'self_attention', 'lr': '0.0001'},
        # {'fusion_type': 'cross_attention', 'lr': '0.0001'},
    ]
    
    # Single model configurations
    single_model_configs = [
        # {'pretrain_model': 'UNI', 'input_dim': '1024', 'lr': '0.001'},
        # {'pretrain_model': 'CHIEF', 'input_dim': '768', 'lr': '0.001'},
        # {'pretrain_model': 'Gigapath', 'input_dim': '768', 'lr': '0.001'},
        # {'pretrain_model': 'Virchow', 'input_dim': '2560', 'lr': '0.001'},
        # {'pretrain_model': 'CONCH', 'input_dim': '768', 'lr': '0.001'},
        # {'pretrain_model': 'PRISM', 'input_dim': '1280', 'lr': '0.001'},
    ]
    
    # Multi-model groups
    model_groups = [
        # {
        #     'base_models': ['CONCH', 'UNI', 'CHIEF_tile'],
        #     'base_model_feature_dims': [768, 1024, 768],
        #     'save_dir': './output'
        # },
        {
            'base_models': ['Gigapath_tile', 'UNI'],
            'base_model_feature_dims': [1536, 1024],
            'save_dir': './output_2'
        }
    ]
    
    # Generate commands for each task
    for task_name, paths in task_configs.items():
        task_cfg_path = paths['task_cfg_path']
        dataset_csv = paths['dataset_csv']
        root_path = paths['root_path']
        
        # Generate fusion model commands
        for model_group in model_groups:
            for fusion_config in fusion_configs:
                cmd = [
                    'python main.py',
                    f'--task_cfg_path "{task_cfg_path}"',
                    f'--dataset_csv "{dataset_csv}"',
                    f'--root_path "{root_path}"',
                    f'--pretrain_model_type "{common_params["pretrain_model_type"]}"',
                    f'--tuning_method "{common_params["tuning_method"]}"',
                    f'--lr "{fusion_config["lr"]}"',
                    f'--batch_size "{common_params["batch_size"]}"',
                    f'--save_dir "{model_group["save_dir"]}"',
                    f'--fusion_type "{fusion_config["fusion_type"]}"',
                    '--base_models ' + ' '.join(f'"{m}"' for m in model_group['base_models']),
                    '--base_model_feature_dims ' + ' '.join(str(d) for d in model_group['base_model_feature_dims'])
                ]
                commands.append(' '.join(cmd))
        
        # Generate single model commands
        for model_config in single_model_configs:
            cmd = [
                'python main.py',
                f'--task_cfg_path "{task_cfg_path}"',
                f'--dataset_csv "{dataset_csv}"',
                f'--root_path "{root_path}"',
                f'--pretrain_model_type "{common_params["pretrain_model_type"]}"',
                f'--tuning_method "{common_params["tuning_method"]}"',
                f'--lr "{model_config["lr"]}"',
                f'--batch_size "{common_params["batch_size"]}"',
                f'--save_dir "./output"',
                '--fusion_type "none"',
                f'--pretrain_model {model_config["pretrain_model"]}',
                f'--input_dim {model_config["input_dim"]}'
            ]
            commands.append(' '.join(cmd))
    
    return commands

# Define your task configurations
task_configs = {
    'BRACS_COARSE': {
        'task_cfg_path': 'task_configs/subtype/BRACS_COARSE.yaml',
        'dataset_csv': 'dataset_csv/subtype/BRACS_COARSE.csv',
        'root_path': '/data4/embedding/BRACS'
    },
    'TCGA-BRCA-SUBTYPE': {
        'task_cfg_path': 'task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml',
        'dataset_csv': 'dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv',
        'root_path': '/data4/embedding/TCGA-BRCA'
    },
    'SLNBREAST_SUBTYPE': {
        'task_cfg_path': 'task_configs/subtype/SLNBREAST_SUBTYPE.yaml',
        'dataset_csv': 'dataset_csv/subtype/SLNBREAST_SUBTYPE.csv',
        'root_path': '/data4/embedding/SLN-Breast'
    },
    'AIDPATH_GRADE': {
        'task_cfg_path': 'task_configs/subtype/AIDPATH_GRADE.yaml',
        'dataset_csv': 'dataset_csv/subtype/AIDPATH_GRADE.csv',
        'root_path': '/data4/embedding/AIDPATH'
    },
    'BCNB_ALN': {
        'task_cfg_path': 'task_configs/subtype/BCNB_ALN.yaml',
        'dataset_csv': 'dataset_csv/subtype/BCNB_ALN.csv',
        'root_path': '/data4/embedding/BCNB'
    },
    'TCGA-BRCA_STAGE': {
        'task_cfg_path': 'task_configs/subtype/TCGA-BRCA_STAGE.yaml',
        'dataset_csv': 'dataset_csv/subtype/TCGA-BRCA_STAGE.csv',
        'root_path': '/data4/embedding/TCGA-BRCA'
    },
    'DORID_6': {
        'task_cfg_path': 'task_configs/subtype/DORID_6.yaml',
        'dataset_csv': 'dataset_csv/subtype/DORID_6.csv',
        'root_path': '/data4/embedding/DORID'
    },
    'BCNB_ER': {
        'task_cfg_path': 'task_configs/biomarker/BCNB_ER.yaml',
        'dataset_csv': 'dataset_csv/biomarker/BCNB_ER.csv',
        'root_path': '/data4/embedding/BCNB'
    },
    'BCNB_PR': {
        'task_cfg_path': 'task_configs/biomarker/BCNB_PR.yaml',
        'dataset_csv': 'dataset_csv/biomarker/BCNB_PR.csv',
        'root_path': '/data4/embedding/BCNB'
    },
    'BCNB_HER2': {
        'task_cfg_path': 'task_configs/biomarker/BCNB_HER2.yaml',
        'dataset_csv': 'dataset_csv/biomarker/BCNB_HER2.csv',
        'root_path': '/data4/embedding/BCNB'
    },
    
}

# Generate all commands
all_commands = generate_commands(task_configs)

# Print or save the commands
for i, cmd in enumerate(all_commands, 1):
    print(f'Command {i}: {cmd}')





commands=all_commands

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
