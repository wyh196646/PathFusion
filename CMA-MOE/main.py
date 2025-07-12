import os
import torch
import pandas as pd
import numpy as np
import sys
from training import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from finetune_utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj, process_predicted_data,process_survival_predicted_data
from datasets.slide_dataset import SlideDataset
import torch.nn as nn
from models.CMA import MultiModelFusionSystem, SingleModelBaseline, SimpleFusionBaseline
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    args = get_finetune_params()


    print(args.root_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device: {}')
    args.device = device
    seed_torch(device, args.seed)
    
    # 选择基础模型数量和token_dim
    if args.fusion_type == 'none':
        use_base_models = [args.pretrain_model]
        num_experts = 1
        # 单模型情况下，取第一个维度
        base_model_dims = [args.input_dim]
    else:
        use_base_models = args.base_models
        num_experts = len(use_base_models)
        # 多模型情况下，使用完整的维度列表
        base_model_dims = getattr(args, 'base_model_feature_dims', [args.fused_feature_dim] * num_experts)
        
    # 验证维度列表长度
    if len(base_model_dims) != num_experts:
        print(f"Warning: base_model_feature_dims length {len(base_model_dims)} doesn't match num_experts {num_experts}")
        base_model_dims = base_model_dims[:num_experts] + [args.fused_feature_dim] * max(0, num_experts - len(base_model_dims))
    
    token_dim = args.fused_feature_dim

    # 实例化模型
    if args.fusion_type == 'none':
        # 单模型基线
        fusion_model = SingleModelBaseline(
            token_dim=token_dim,
            n_classes=2,  # 这个会在后面根据任务配置更新
            head_type="mil",  # 使用MIL头
            base_model_feature_dims=base_model_dims
        ).to(args.device)
    elif args.fusion_type in ['concat', 'self_attention', 'cross_attention']:
        # 简单融合基线
        fusion_model = SimpleFusionBaseline(
            fusion_type=args.fusion_type,
            num_experts=num_experts,
            token_dim=token_dim,
            n_classes=2,  # 这个会在后面根据任务配置更新
            mlp_hidden=getattr(args, "mlp_hidden", 256),
            attn_heads=getattr(args, "attention_heads", 8),
            base_model_feature_dims=base_model_dims
        ).to(args.device)
    elif args.fusion_type == 'moe':
        # 完整的多模型融合系统
        fusion_model = MultiModelFusionSystem(
            fusion_type=args.fusion_type,
            num_experts=num_experts,
            token_dim=token_dim,
            num_tokens=64,  
            n_classes=2,  
            memory_size=getattr(args, "memory_slots", 32),
            num_memory_layers=getattr(args, "num_memory_layers", 2),
            gamma=getattr(args, "gamma", 0.1),
            mlp_hidden=getattr(args, "mlp_hidden", 256),
            attn_heads=getattr(args, "attention_heads", 8),
            base_model_feature_dims=base_model_dims
        ).to(args.device)
    else:
        raise ValueError(f"Unknown fusion_type: {args.fusion_type}")
        
    print(f"Created model with fusion_type: {args.fusion_type}, num_experts: {num_experts}")
    print(f"Base model feature dimensions: {base_model_dims}")
    print(f"Model parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)}")


    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config= load_task_config(args.task_cfg_path)
    label_dict = args.task_config.get('label_dict', None)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    args.task_type = args.task_config.get('setting', 'classification')
    # set the experiment save directory
    if args.fusion_type == 'none':
        args.save_dir = os.path.join(args.save_dir, args.task, args.pretrain_model, args.tuning_method, str(args.lr))
    else:
        args.save_dir = os.path.join(args.save_dir, args.task, args.fusion_type, args.tuning_method, str(args.lr))

    args.model_code, args.task_code, args.exp_code, = get_exp_code(args) # get the experiment code
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.split_key = 'pat_id'
    else:
        args.split_key = 'slide_id'

    # set up the dataset
    args.split_dir = os.path.join(args.split_dir,args.task_code) if not args.pre_split_dir else args.pre_split_dir
    os.makedirs(args.split_dir, exist_ok=True)
    print('Setting split directory: {}'.format(args.split_dir))
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file
        
    prediction_save_dir = os.path.join(args.save_dir,'prediction_results')
    prediction_file = os.path.join(prediction_save_dir, 'val_predict.csv')
    if os.path.exists(prediction_file):
        #jump this exucution
        print(f"Prediction file {prediction_file} already exists. Skipping execution.")
        sys.exit(0)
    os.makedirs(prediction_save_dir,exist_ok=True)
    
    
    DatasetClass = SlideDataset
    
    
    

            
    results = {}
    predict_results ={}
    # start cross validation

    for fold in range(args.folds):
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}')

        train_splits, val_splits, test_splits = get_splits(dataset, fold=fold, **vars(args))
        # instantiate the dataset
        train_data, val_data, test_data = DatasetClass(dataset, 
                                                        args.root_path, 
                                                        train_splits,
                                                        args.task_config, 
                                                        split_key=args.split_key,
                                                        base_models=use_base_models
                                                        ) \
                                        , DatasetClass(dataset,
                                                        args.root_path, 
                                                        val_splits,
                                                        args.task_config, 
                                                        split_key=args.split_key if len(val_splits) > 0 else None,
                                                        base_models=use_base_models
                                                       ) \
                                        , DatasetClass(dataset,
                                                        args.root_path,
                                                        test_splits, args.task_config, 
                                                        split_key=args.split_key if len(test_splits) > 0 else None,
                                                        base_models=use_base_models
                                                        ) 
        
        args.n_classes = train_data.n_classes # get the number of classes
        
        # Update model with correct number of classes
        if hasattr(fusion_model, 'mil_head'):
            # Update the final layer to match n_classes
            final_layer = fusion_model.mil_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.out_features != args.n_classes:
                print(f"Updating model output layer from {final_layer.out_features} to {args.n_classes} classes")
                fusion_model.mil_head[-1] = nn.Linear(final_layer.in_features, args.n_classes).to(args.device)
        elif hasattr(fusion_model, 'head'):
            # For SimpleFusionBaseline
            final_layer = fusion_model.head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.out_features != args.n_classes:
                print(f"Updating model output layer from {final_layer.out_features} to {args.n_classes} classes")
                fusion_model.head[-1] = nn.Linear(final_layer.in_features, args.n_classes).to(args.device)


        fusion_model.n_classes = args.n_classes  

        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        

# start training

        val_records, test_records = train(
            (train_loader, val_loader, test_loader), 
            fold, 
            args,
            fusion_model=fusion_model
        )

        # update the results
        
        records = {'val': val_records, 'test': test_records}
        for record_ in records:
            for key in records[record_]:
                key_ = record_ + '_' + key
                if 'prob' in key or 'label' in key or 'slide_id' in key:
                    if key_ not in predict_results:
                        predict_results[key_] = []
                    predict_results[key_].append(records[record_][key])
                else:   
                    if key_ not in results:
                        results[key_] = []
                    results[key_].append(records[record_][ key])
    
    if args.task_type != 'survival':
        val_predict_df = process_predicted_data(predict_results, label_dict.keys(), 'val')
        val_predict_df.to_csv(os.path.join(prediction_save_dir,'val_predict.csv'),index=True)
        
        test_predict_df = process_predicted_data(predict_results,label_dict.keys(),'test')
        test_predict_df.to_csv(os.path.join(prediction_save_dir,'test_predict.csv'),index=True)
    else:
        val_df = process_survival_predicted_data(predict_results, section='val')
        val_df.to_csv(os.path.join(prediction_save_dir, 'val_survival_prediction.csv'), index=True)
        test_df = process_survival_predicted_data(predict_results, section='test')
        test_df.to_csv(os.path.join(prediction_save_dir, 'test_survival_prediction.csv'), index=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')



#CUDA_VISIBLE_DEVICES=  python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "moe"
#python main.py --task_cfg_path "task_configs/subtype/BRACS_COARSE.yaml" --dataset_csv "dataset_csv/subtype/BRACS_COARSE.csv" --root_path "/data4/embedding/BRACS" --pretrain_model_type "slide_level" --tuning_method "MIL" --lr "0.0001" --batch_size "1" --save_dir "./output" --fusion_type "none" --pretrain_model UNI --input_dim 1024