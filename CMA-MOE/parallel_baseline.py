import os
import subprocess
import time

# 用户可配置的 GPU 列表及最大任务数
gpu_config = {
    0: 16,  
    1: 16, 
    2: 16, 
    3: 16,  
    4: 16,
    5: 16,  
    6: 16,   
    7: 16,
}
# pretrain_models = ['Gigapath_tile', 'CONCH', 'UNI', 'TITAN', 'Virchow',
#                    'CHIEF_tile', 'Gigapath', 'CHIEF','FMBC','PRISM',
#                    'FMBC_Slide_25_cls','FMBC_Slide_75_cls','FMBC_Slide_50_cls',
#                    'FMBC_Slide_100_cls','UNI_Slide_25_cls','Gigapath_tile_Slide_25_cls']
pretrain_models = ['Gigapath_tile', 'CONCH', 'UNI', 
                   'CHIEF_tile', 'Gigapath', 'CHIEF','PRISM',
                   'FMBC_Slide_25_cls']
pretrain_model_dim_dict = {
    "UNI": 1024,
    "CONCH": 768,
    "CHIEF_tile": 768,
    "TITAN": 768,
    "Virchow": 2560,
    "Gigapath_tile": 1536,
    "Gigapath": 768,
    "CHIEF": 768,
    'FMBC':768,
    'PRISM':1280,
    'FMBC_Slide_25':768,
    'FMBC_Slide_25_cls':768,
    'FMBC_Slide_75_cls':768,
    'FMBC_Slide_100':768,
    'FMBC_Slide_100_cls':768,
    'UNI_Slide_25_cls':768,
    'Gigapath_tile_Slide_25_cls':768,

}
pretrain_model_types_dict = {
    "UNI": "patch_level",
    "CONCH": "patch_level",
    "CHIEF_tile": "patch_level",
    "TITAN": "slide_level",
    "Virchow": "patch_level",
    "Gigapath_tile": "patch_level",
    "Gigapath": "slide_level",
    "CHIEF": "slide_level",
    'FMBC': 'slide_level',
    'PRISM':'slide_level',
    'FMBC_Slide_25':'slide_level',
    'FMBC_Slide_25_cls':'slide_level',
    'FMBC_Slide_75_cls':'slide_level',
    'FMBC_Slide_100':'slide_level',
    'FMBC_Slide_100_cls':'slide_level',
    'UNI_Slide_25_cls':'slide_level',
    'Gigapath_tile_Slide_25_cls':'slide_level'
}

def get_tuning_methods(pretrain_model,model_type):
    if pretrain_model =='FMBC':
        return ["LR_Same_Patch"]
    # 如果模型类型为patch_level，则返回["LR"]，否则返回["LR"]
    return ["LR"] if model_type == "patch_level" else ["LR"]#,"ABMIL"

learning_rates = [0.001, 0.0001]#


def get_available_gpus():
    return list(gpu_config.keys())

def is_task_hanging(process):
    # 运行nvidia-smi命令，查询计算应用的pid和使用的内存，并以csv格式输出，不显示表头
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], 
                            capture_output=True, text=True)
    # 获取命令输出结果
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
commands_file = "baseline.txt"
if os.path.exists(commands_file):
    os.remove(commands_file)
import yaml
tasks = yaml.load(open('all_task.yaml', 'r'), Loader=yaml.FullLoader)
with open(commands_file, "w") as f:    
    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        task_cfg = config["task_cfg"]
        #if task contain key fold
        if "folds" in config.keys():
            folds = config['folds']
        else:
            folds = 5
        if "batch_size" in config.keys():
            batch_size = config['batch_size']
        else:
            batch_size = 1
        for pretrain_model in pretrain_models:
            pretrain_model_type = pretrain_model_types_dict[pretrain_model]
            tuning_methods = get_tuning_methods(pretrain_model, pretrain_model_type)
            input_dim = pretrain_model_dim_dict[pretrain_model]
            root_path = os.path.join(embedding_dir, pretrain_model)
            dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")


            for tuning_method in tuning_methods:
                for learning_rate in learning_rates:
                    output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, str(learning_rate),'summary.csv')
                    
                    if os.path.exists(output_prediction):
                        print(f"Skipping task: {output_prediction} already exists")
                        continue
                    command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                            f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                            f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method} --lr {learning_rate} "\
                            f"--folds {folds} --batch_size {batch_size}"
                    task_queue.append((task_name, command))
                    save_command = command+'\n'
                    f.write(save_command)
    while task_queue or any(len(v) > 0 for v in running_tasks.values()):
        for gpu in gpu_list:
            running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
            if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
                task_name, cmd = task_queue.pop(0)
                run_task(task_name, cmd, gpu)
        
        time.sleep(10)

    print("All tasks completed.")
