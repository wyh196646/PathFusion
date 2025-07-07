import os
import sys
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
os.environ["WANDB_API_KEY"] = '6ebb1c769075243eb73a32c4f9f7011ddd41f20a'

import torch
import numpy as np
import torch.utils.tensorboard as tensorboard

from metrics import calculate_metrics_with_task_cfg
from finetune_utils import (
    get_optimizer, get_loss_function, Monitor_Score, get_records_array,
    log_writer, adjust_learning_rate, release_nested_dict,
    initiate_mil_model, initiate_linear_model
)

def train(dataloader, fold, args, fusion_model=None):
    train_loader, val_loader, test_loader = dataloader
    # set up the writer
    writer_dir = os.path.join(args.save_dir, f'fold_{fold}', 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

        
    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project= args.task,
            name=args.fusion_type + '_fold_' + str(fold)+ '_'+ str(args.tuning_method)+'_'+str(args.lr),
            id='fold_' + str(fold)+'_'+str(args.pretrain_model)+'_'+  str(args.tuning_method)+'_'+str(args.lr),
            tags=[],
            config=vars(args),
            settings=wandb.Settings(init_timeout=120)
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)

    # 决定模型
    if args.fusion_type == 'none':
        model = initiate_linear_model(args)  # 单模型，线性头
        fusion = None
    else:
        model = fusion_model  # 多模型融合
        fusion = fusion_model

    model = model.to(args.device)
    optimizer = get_optimizer(args, model)
    loss_fn = get_loss_function(args.task_config)
    monitor = Monitor_Score()
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset)) if val_loader is not None else None)
    print('Testing on {} samples'.format(len(test_loader.dataset)) if test_loader is not None else None)
    print('Training starts!')

    val_records, test_records = None, None
    if os.path.exists(os.path.join(args.save_dir, f'fold_{fold}', 'checkpoint.pt')) :
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")))
        test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, 0, args, fusion)
        val_records = test_records
        print(f'Fold {fold} already exists, skipping...')
        print('################################')
        print('################################')
        print('################################')
        return val_records, test_records

    for i in range(args.epochs):
        print('Epoch: {}'.format(i))
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args, fusion)

        if val_loader is not None:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args, fusion)

            # update the writer for train and val
            log_dict = {'train_' + k: v for k, v in train_records.items() if 'prob' not in k and 'label' not in k}
            log_dict.update({'val_' + k: v for k, v in val_records.items() if 'prob' not in k and 'label' not in k})
            log_writer(log_dict, i, args.report_to, writer)
            # update the monitor scores
            task_setting = args.task_config.get('setting', 'multi_class')
            if task_setting == 'regression':
                scores = val_records['mae']
            elif task_setting == 'binary_classification' or task_setting == 'multi_class':
                scores = val_records['bacc']
            elif task_setting == 'survival':
                scores = val_records['c_index']
            else:
                scores = val_records['macro_auroc']
                
        if args.model_select == 'val' and val_loader is not None:
            monitor(scores, model, ckpt_name=os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))
        elif args.model_select == 'last_epoch' and i == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))

    # load model for test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")))
    # test the model
    test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, i, args, fusion)
    # update the writer for test
    log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
    log_writer(log_dict, fold, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args, fusion=None):
    model.train()
    # set the start time
    start_time = time.time()
    import pickle

    seq_len = 0

    # setup the records
    records = get_records_array(len(train_loader), args.n_classes)

    for batch_idx, batch in enumerate(train_loader):
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            # 判断输入结构
            if args.fusion_type == 'none':
                # 单模型：[B, T, D]
                images, img_coords, pad_mask, label = batch['imgs'], batch['coords'], batch['pad_mask'], batch['labels']
                images = images.to(args.device, non_blocking=True)
                img_coords = img_coords.to(args.device, non_blocking=True)
                pad_mask = pad_mask.to(args.device, non_blocking=True)
                label = label.to(args.device, non_blocking=True)
                
                # Debug input
                if torch.isnan(images).any():
                    print(f"NaN detected in images at batch {batch_idx}")
                    images = torch.nan_to_num(images, nan=0.0)
                
                logits = model(images, img_coords, pad_mask)
            else:
                # 多模型：[B, T, D] * n_models
                imgs_list = [x.to(args.device, non_blocking=True) for x in batch['imgs_list']]
                coords_list = [x.to(args.device, non_blocking=True) for x in batch['coords_list']]
                pad_mask_list = [x.to(args.device, non_blocking=True) for x in batch['pad_mask_list']]
                label = batch['labels'].to(args.device, non_blocking=True)
                
                # Debug inputs
                for i, imgs in enumerate(imgs_list):
                    if torch.isnan(imgs).any():
                        print(f"NaN detected in imgs_list[{i}] at batch {batch_idx}")
                        imgs_list[i] = torch.nan_to_num(imgs, nan=0.0)
                
                logits = fusion(imgs_list, coords_list, pad_mask_list)
            
            # Validate and debug logits and labels
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"NaN/Inf detected in logits at batch {batch_idx}")
                print(f"Logits shape: {logits.shape}, stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            # Prepare labels based on loss function type
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
                # Ensure labels are in [0, 1] range
                label = torch.clamp(label, 0, 1)
            elif isinstance(loss_fn, torch.nn.MSELoss):
                label = label.squeeze(-1).float()
            else:  # CrossEntropyLoss
                label = label.squeeze(-1).long()
                # Critical: Ensure labels are in valid range [0, n_classes-1]
                max_class = args.n_classes - 1
                if label.max() > max_class or label.min() < 0:
                    print(f"ERROR: Invalid label range at batch {batch_idx}")
                    print(f"Label stats: min={label.min().item()}, max={label.max().item()}, expected range=[0, {max_class}]")
                    print(f"Labels: {label}")
                    print(f"Logits shape: {logits.shape}, n_classes: {args.n_classes}")
                    # Clamp labels to valid range
                    label = torch.clamp(label, 0, max_class)
                
                # Ensure logits have correct number of classes
                if logits.shape[-1] != args.n_classes:
                    print(f"ERROR: Logits dimension mismatch at batch {batch_idx}")
                    print(f"Logits shape: {logits.shape}, expected n_classes: {args.n_classes}")
                    continue  # Skip this batch

            # Additional debugging for CrossEntropyLoss
            # if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            #     print(f"Batch {batch_idx}: logits shape={logits.shape}, labels shape={label.shape}")
            #     print(f"Label range: [{label.min().item()}, {label.max().item()}], n_classes: {args.n_classes}")
                
            loss = loss_fn(logits, label)
            
            # Debug loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss detected at batch {batch_idx}")
                print(f"Logits: {logits}")
                print(f"Labels: {label}")
                print(f"Loss function: {type(loss_fn)}")
                continue  # Skip this batch
            
            loss /= args.gc

            if fp16_scaler is None:
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                fp16_scaler.scale(loss).backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    # Add gradient clipping
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                    optimizer.zero_grad()

        # update the records
        records['loss'] += loss.item() * args.gc

        if (batch_idx + 1) % 20 == 0:
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.4f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss']/batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss))
    return records

def chunk_batch(batch, chunk_size):
    return [batch[k:k+chunk_size] for k in range(0, len(batch), chunk_size)]

def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, fusion=None):
    model.eval()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes)

    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    if task_setting == 'survival':
        records = {
            'prob': [],
            'label': [],
            'slide_id': [],
            'loss': 0.0,
        }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.fusion_type == 'none':
                images, img_coords, pad_mask, label, slide_id = batch['imgs'], batch['coords'], batch['pad_mask'], batch['labels'], batch['slide_id']
                images = images.to(args.device, non_blocking=True)
                img_coords = img_coords.to(args.device, non_blocking=True)
                pad_mask = pad_mask.to(args.device, non_blocking=True)
                label = label.to(args.device, non_blocking=True)
                logits = model(images, img_coords, pad_mask)
            else:
                imgs_list = [x.to(args.device, non_blocking=True) for x in batch['imgs_list']]
                coords_list = [x.to(args.device, non_blocking=True) for x in batch['coords_list']]
                pad_mask_list = [x.to(args.device, non_blocking=True) for x in batch['pad_mask_list']]
                label = batch['labels'].to(args.device, non_blocking=True)
                slide_id = batch['slide_id']
                
                logits = fusion(imgs_list, coords_list, pad_mask_list)

            # Prepare labels with validation
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
                label = torch.clamp(label, 0, 1)
            else:
                label = label.squeeze(-1)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    label = label.long()
                    # Validate label range
                    max_class = args.n_classes - 1
                    if label.max() > max_class or label.min() < 0:
                        print(f"ERROR: Invalid label range in evaluation at batch {batch_idx}")
                        print(f"Label stats: min={label.min().item()}, max={label.max().item()}, expected range=[0, {max_class}]")
                        label = torch.clamp(label, 0, max_class)
                        
            loss = loss_fn(logits, label)

            # update the records
            records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'][batch_idx] = Y_prob.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
                records['slide_id'].extend(slide_id)
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu()
                records['prob'][batch_idx] = Y_prob.numpy()
                # convert label to one-hot
                label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                records['label'][batch_idx] = label_.numpy()
                records['slide_id'].extend(slide_id)
            elif task_setting == 'regression':
                records['prob'][batch_idx] = logits.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
                records['slide_id'].extend(slide_id)
            elif task_setting == 'survival':
                records['prob'].extend( logits.squeeze().cpu().numpy())
                records['label'].extend(label.cpu().numpy())
                records['slide_id'].extend(slide_id)
    records.update(release_nested_dict(calculate_metrics_with_task_cfg(records['prob'], 
                                                                       records['label'], 
                                                                       args.task_config)))
    
    records['loss'] = records['loss'] / len(loader)

    if task_setting == 'multi_label':
        info = 'Epoch: {}, Loss: {:.4f}, Micro AUROC: {:.4f}, Macro AUROC: {:.4f}, Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}'.format(epoch, records['loss'], records['micro_auroc'], records['macro_auroc'], records['micro_auprc'], records['macro_auprc'])
    elif task_setting =='regression':
        info = 'Epoch: {}, Loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Pearson: {:.4f}, spearman: {:.4f}'.format(epoch, records['loss'], records['mae'], records['mse'], records['rmse'], records['average_pearson'], records['average_spearman'])
    elif task_setting == 'survival':
        info = 'Epoch: {}, Loss: {:.4f}, C-Index: {:.4f}'.format(epoch, records['loss'], records['c_index'])
    else:
        info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'], records['macro_auroc'], records['acc'], records['bacc'])
        for metric in args.task_config.get('add_metrics', []):
            info += ', {}: {:.4f}'.format(metric, records[metric])

    print(info)
    return records


