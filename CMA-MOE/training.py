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
import torch.nn as nn
import numpy as np
import torch.utils.tensorboard as tensorboard
import torch.nn.functional as F
from metrics import calculate_metrics_with_task_cfg
from finetune_utils import (
    get_optimizer, get_loss_function, Monitor_Score, get_records_array,
    log_writer, adjust_learning_rate, release_nested_dict,
    initiate_mil_model, initiate_linear_model
)

class CrossExpertMemoryContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, expert_outputs):
        M = len(expert_outputs)
        if M <= 1:
            return torch.tensor(0.0, device=expert_outputs[0].device, requires_grad=True)

        loss = 0
        for m in range(M):
            pos = expert_outputs[m]
            neg_list = [expert_outputs[n] for n in range(M) if n != m]
            if len(neg_list) == 0:
                continue  # 增加保护逻辑
            neg = torch.cat(neg_list, dim=0)
            pos_sim = torch.exp(torch.sum(pos * pos, dim=-1) / self.temperature)
            neg_sim = torch.exp(torch.mm(pos, neg.T) / self.temperature).sum(dim=-1)
            loss += -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()

        loss = loss / M
        return loss


class AdaptiveInfoPreservationLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pooled_features, positive_features):
        # pooled_features: [B, D]
        # positive_features: [B, D]
        B = pooled_features.size(0)

        pooled_features_norm = F.normalize(pooled_features, dim=-1)
        positive_features_norm = F.normalize(positive_features, dim=-1)

        sim_matrix = torch.matmul(pooled_features_norm, positive_features_norm.T) / self.temperature
        labels = torch.arange(B, device=pooled_features.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss



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
        fp16_scaler = torch.amp.GradScaler('cuda')
        print('Using fp16 training')

    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset)) if val_loader is not None else None)
    print('Testing on {} samples'.format(len(test_loader.dataset)) if test_loader is not None else None)
    print('Training starts!')

    val_records, test_records = None, None
    # if os.path.exists(os.path.join(args.save_dir, f'fold_{fold}', 'checkpoint.pt')) :

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

    # Initialize loss function components
    lambda_param = torch.tensor(0.1, requires_grad=True, device=args.device)
    mu_param = torch.tensor(0.1, requires_grad=True, device=args.device)
    cemcl_loss_fn = CrossExpertMemoryContrastiveLoss()
    info_loss_fn = AdaptiveInfoPreservationLoss(args.input_dim if hasattr(args, 'input_dim') else 512)

    for batch_idx, batch in enumerate(train_loader):
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        with torch.amp.autocast('cuda', dtype=torch.float16 if args.fp16 else torch.float32):
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
                
                imgs_list = [x.to(args.device, non_blocking=True) for x in batch['imgs_list']]
                coords_list = [x.to(args.device, non_blocking=True) for x in batch['coords_list']]
                pad_mask_list = [x.to(args.device, non_blocking=True) for x in batch['pad_mask_list']]
                label = batch['labels'].to(args.device, non_blocking=True)
                

                
                logits = fusion(imgs_list, coords_list, pad_mask_list)
                images = imgs_list[0] if imgs_list else torch.zeros(1, 1, 1, device=args.device)
            
            # Validate and debug logits and labels


            # Prepare labels based on loss function type - handle batch dimension
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()  # [B]
                label = torch.clamp(label, 0, 1)
            elif isinstance(loss_fn, torch.nn.MSELoss):
                label = label.squeeze(-1).float()  # [B]
            else:  # CrossEntropyLoss
                label = label.squeeze(-1).long()  # [B]
                # Critical: Ensure labels are in valid range [0, n_classes-1]
                max_class = args.n_classes - 1
                if label.max() > max_class or label.min() < 0:
                    print(f"ERROR: Invalid label range at batch {batch_idx}")
                    print(f"Label stats: min={label.min().item()}, max={label.max().item()}, expected range=[0, {max_class}]")
                    print(f"Labels: {label}")
                    print(f"Logits shape: {logits.shape}, n_classes: {args.n_classes}")
                    label = torch.clamp(label, 0, max_class)
                

            if args.fusion_type != 'moe':
                #logits = fusion(imgs_list, coords_list, pad_mask_list)
                expert_outputs = [logits]
                pooled_features = logits
                raw_pooled_features = logits.detach()
            else:
                logits, expert_outputs, pooled_features, raw_pooled_features = fusion(imgs_list, coords_list, pad_mask_list)
           
            classification_loss = loss_fn(logits, label)
            cemcl_loss = cemcl_loss_fn(expert_outputs)

            # 修复后的调用方式：使用raw_pooled_features作为正样本特征
            info_loss = info_loss_fn(pooled_features, raw_pooled_features.detach())


            
            loss = classification_loss + lambda_param * cemcl_loss #+ mu_param * info_loss
            
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
            
            if (batch_idx + 1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            fp16_scaler.scale(loss).backward()
            
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
                  .format(epoch, batch_idx, records['loss']/(batch_idx+1), optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len/(batch_idx+1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

    records['loss'] = records['loss'] / len(train_loader)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, records['loss']))
    return records

def chunk_batch(batch, chunk_size):
    return [batch[k:k+chunk_size] for k in range(0, len(batch), chunk_size)]

def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, fusion=None):
    model.eval()

    # set the evaluation records - adjust for batch processing
    all_probs = []
    all_labels = []
    all_slide_ids = []
    total_loss = 0.0
    num_batches = 0

    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    
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
                
                # Handle potential tuple return from fusion model
                fusion_output = fusion(imgs_list, coords_list, pad_mask_list)
                if isinstance(fusion_output, tuple):
                    logits = fusion_output[0]  # Take the first element (logits)
                else:
                    logits = fusion_output

            # Prepare labels with validation - handle batch dimension
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
                label = torch.clamp(label, 0, 1)
            else:
                label = label.squeeze(-1)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    label = label.long()
                    max_class = args.n_classes - 1
                    if label.max() > max_class or label.min() < 0:
                        print(f"ERROR: Invalid label range in evaluation at batch {batch_idx}")
                        print(f"Label stats: min={label.min().item()}, max={label.max().item()}, expected range=[0, {max_class}]")
                        label = torch.clamp(label, 0, max_class)
                        
            loss = loss_fn(logits, label)
            total_loss += loss.item()
            num_batches += 1

            # Process predictions and labels for metrics - handle batch dimension
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits).cpu().numpy()  # [B, n_classes]
                all_probs.append(Y_prob)
                all_labels.append(label.cpu().numpy())  # [B, n_classes]
                all_slide_ids.extend(slide_id)
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu().numpy()  # [B, n_classes]
                all_probs.append(Y_prob)
                # convert label to one-hot
                label_onehot = torch.zeros(label.size(0), args.n_classes)  # [B, n_classes]
                label_onehot.scatter_(1, label.cpu().unsqueeze(1), 1)
                all_labels.append(label_onehot.numpy())
                all_slide_ids.extend(slide_id)
            elif task_setting == 'regression':
                all_probs.append(logits.cpu().numpy())  # [B, 1]
                all_labels.append(label.cpu().numpy())  # [B, 1]
                all_slide_ids.extend(slide_id)
            elif task_setting == 'survival':
                all_probs.extend(logits.squeeze().cpu().numpy())  # [B] -> extend to list
                all_labels.extend(label.cpu().numpy())            # [B, 2] -> extend to list
                all_slide_ids.extend(slide_id)
    
    # Concatenate all batches
    if task_setting != 'survival':
        all_probs = np.vstack(all_probs)      # [total_samples, n_classes]
        all_labels = np.vstack(all_labels)    # [total_samples, n_classes]
    
    # Create records for metric calculation
    records = {
        'prob': all_probs,
        'label': all_labels,
        'slide_id': all_slide_ids,
        'loss': total_loss / num_batches
    }
    
    # Calculate metrics
    records.update(release_nested_dict(calculate_metrics_with_task_cfg(records['prob'], 
                                                                       records['label'], 
                                                                       args.task_config)))
    
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


