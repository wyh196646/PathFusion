import os
import subprocess
import time
import yaml

# GPU 配置（使用更高并发配置）
gpu_config = {i: 6 for i in range(8)}

pretrain_models = ['FMBC']
pretrain_model_dim_dict = {"FMBC": 1024}
pretrain_model_types_dict = {"FMBC": "slide_level"}

# 多模型配置
ablation_models = {
    'UNI': '/home/yuhaowang/project/FMBC/SlideModel/slide_UNI/checkpoint0090.pth',
    'Gigapath': '/home/yuhaowang/project/FMBC/SlideModel/slide_gigapath/checkpoint0090.pth',
}

learning_rates = [0.001, 0.0001]

def get_tuning_methods(pretrain_model):
    if pretrain_model == 'FMBC':
        return [f"LR_{lr}_{pool}" for lr in ['Same_0.25'] for pool in ['MeanPool']]
    return []

def get_available_gpus():
    return list(gpu_config.keys())

def is_task_hanging(process):
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    return str(process.pid) in output and "0 MiB" in output

def run_task(task_name, command, gpu_id):
    print(f"Starting: {command} on GPU {gpu_id}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, shell=True, env=env)
    running_tasks[gpu_id].append((process, command))

# 加载任务
tasks = yaml.load(open('all_task.yaml', 'r'), Loader=yaml.FullLoader)

gpu_list = get_available_gpus()
running_tasks = {gpu: [] for gpu in gpu_list}
task_queue = []

# 任务组合构建
for task_name, config in tasks.items():
    embedding_dir = config["embedding_dir"]
    csv_dir = config["csv_dir"]
    task_cfg = config["task_cfg"]
    batch_size = config.get("batch_size", 1)

    for ablation_model, slide_weight_path in ablation_models.items():
        for pretrain_model in pretrain_models:
            pretrain_model_type = pretrain_model_types_dict[pretrain_model]
            tuning_methods = get_tuning_methods(pretrain_model)
            input_dim = pretrain_model_dim_dict[pretrain_model]
            root_path = os.path.join(embedding_dir, pretrain_model)
            dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")

            for tuning_method in tuning_methods:
                for learning_rate in learning_rates:
                    # 生成输出路径并检查是否已存在
                    output_prediction = os.path.join(
                        'outputs', task_name, pretrain_model, tuning_method,
                        str(learning_rate), 'summary.csv'
                    ).replace('FMBC', f"{'Gigapath_tile' if ablation_model == 'Gigapath' else 'UNI'}")

                    if os.path.exists(output_prediction):
                        print(f"Skipping task: {output_prediction} already exists")
                        continue

                    command = (
                        f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} "
                        f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} "
                        f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method} "
                        f"--lr {learning_rate} --pretrained {slide_weight_path} --batch_size {batch_size} "
                        f"--ablation_model {ablation_model}"
                    )

                    task_queue.append((task_name, command))

# 启动调度
while task_queue or any(len(v) > 0 for v in running_tasks.values()):
    for gpu in gpu_list:
        running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
        if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
            task_name, cmd = task_queue.pop(0)
            run_task(task_name, cmd, gpu)

    time.sleep(10)

print("All tasks completed.")
