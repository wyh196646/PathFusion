

import os
import subprocess
import time
gpu_config = {
    0: 2,
    1: 2, 
    2: 2,  
    3: 3,  
    4: 3, 
    5: 3, 
    6: 3,
    7: 3,
}
pretrain_models = ['FMBC']
pretrain_model_dim_dict = {

    "FMBC":768,
}
pretrain_model_types_dict = {

    "FMBC": "slide_level"
}
def get_tuning_methods(pretrain_model):

    if pretrain_model =='FMBC':
        combinations = [
            f"LR_{lr}_{pool}"
            for lr in [ 'Same_0.25']#"Frozen","Same", "Different",,'Same_1'
            for pool in ["MeanPool" ]#"CLSPool"
        ]
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


task_queue = []
slide_weight_path= '/home/yuhaowang/project/FMBC/Weights/slide/train_from_our_FMBC/checkpoint0160.pth'
import yaml
tasks = yaml.load(open('all_task.yaml', 'r'), Loader=yaml.FullLoader)
learning_rates = [0.001, 0.0001]#
for task_name, config in reversed(list(tasks.items())):#
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
                
                if os.path.exists(output_prediction):
                    print(f"Skipping task: {output_prediction} already exists")
                    continue
                
                command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                        f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                        f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method} --lr {learning_rate} --pretrained {slide_weight_path}\
                            --batch_size {batch_size}"
                task_queue.append((task_name, command))

while task_queue or any(len(v) > 0 for v in running_tasks.values()):
    for gpu in gpu_list:
        running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
        if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
            task_name, cmd = task_queue.pop(0)
            run_task(task_name, cmd, gpu)
    
    time.sleep(10)

print("All tasks completed.")


