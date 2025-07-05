import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             balanced_accuracy_score, accuracy_score, 
                             cohen_kappa_score, mean_absolute_error,
                             mean_squared_error, f1_score)
from sksurv.metrics import concordance_index_censored
import numpy as np
import seaborn as sns
import os
import json
import torch
sns.set_style("white")
# -------------------------------
# Text Generation Metric Imports
# -------------------------------
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# -------------------------------------------------
# Original compute_scores function (from snippet)
# -------------------------------------------------
def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation 
    (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the IDs and their gold captions
    :param res: Dictionary with the IDs and their generated captions
    :return: A dictionary of metrics, e.g. {"BLEU_1": x, "BLEU_2": y, ...}
    """
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        # Some metrics return a list of scores (e.g., BLEU) 
        # or a single score (e.g., METEOR, ROUGE).
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

# -------------------------------------------------
# Original MakeMetrics class (no changes)
# -------------------------------------------------
class MakeMetrics:
    '''
    A class to calculate metrics for multilabel classification and regression tasks.
    
    Arguments:
    ----------
    metric (str): the metric to calculate. Default is 'auroc'. 
                  Options are 'auroc', 'auprc', 'bacc', 'acc', 'qwk', 
                  'pearson', 'spearman', 'mae', 'mse', 'rmse', 'weighted_f1', 'c_index'.
    average (str): the averaging strategy. Default is 'micro'.
    label_dict (dict): the label dictionary, mapping from label to index. Default is None. 
    '''
    def __init__(self, metric='auroc', average='micro', label_dict: dict=None):
        self.metric = metric
        self.average = average
        self.label_dict = label_dict
        self.reversed_dict = {v: k for k, v in label_dict.items()}

    def get_metric(self, labels: np.array, probs: np.array):
        '''Return the metric score based on the metric name.'''
        if self.metric == 'auroc':
            return roc_auc_score(labels, probs, average=self.average)
        elif self.metric == 'auprc':
            return average_precision_score(labels, probs, average=self.average)
        elif self.metric == 'bacc':
            return balanced_accuracy_score(labels, probs)
        elif self.metric == 'acc':
            return accuracy_score(labels, probs)
        elif self.metric == 'qwk':
            return cohen_kappa_score(labels, probs, weights='quadratic')
        elif self.metric == 'weighted_f1':
            probs = probs > 0.5
            return f1_score(labels, probs, average='weighted')
        elif self.metric == 'mae':
            return mean_absolute_error(labels, probs)
        elif self.metric == 'mse':
            return mean_squared_error(labels, probs)
        elif self.metric == 'rmse':
            return mean_squared_error(labels, probs)
        elif self.metric == 'pearson':
            evaluation_results = {
                f"pearson_{self.reversed_dict.get(i)}": pearsonr(labels[:, i], probs[:, i])[0]
                for i in range(labels.shape[1])
            }
            evaluation_results['average_pearson'] = np.mean(list(evaluation_results.values()))
            return evaluation_results
        elif self.metric == 'spearman':
            evaluation_results = {
                f"spearman_{self.reversed_dict.get(i)}": spearmanr(labels[:, i], probs[:, i])[0] 
                for i in range(labels.shape[1])
            }
            evaluation_results['average_spearman'] = np.mean(list(evaluation_results.values()))
            return evaluation_results
        elif self.metric == 'c_index':
            labels =np.array(labels)
            event_times = labels[:, 0]
            censor_flags = labels[:, 1]  
            c_index_value = concordance_index_censored(
                event_indicator=(1 - censor_flags).astype(bool),
                event_time=event_times,
                estimate=probs,
                tied_tol=1e-08
            )[0]
            return c_index_value
        
        else:
            raise ValueError('Invalid metric: {}'.format(self.metric))
        
    def process_preds(self, labels: np.array, probs: np.array):
        '''Process the predictions and labels for classification tasks.'''
        if self.metric in ['bacc', 'acc', 'qwk']:
            return np.argmax(labels, axis=1), np.argmax(probs, axis=1)
        else:
            return labels, probs
    
    @property
    def get_metric_name(self):
        '''Return the metric name.'''
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return '{}_{}'.format(self.average, self.metric)
            else:
                label_keys = sorted(self.label_dict.keys(), key=lambda x: self.label_dict[x])
                return ['{}_{}'.format(key, self.metric) for key in label_keys]
        else:
            return self.metric
        
    def __call__(self, labels: np.array, probs: np.array) -> dict:
        '''
        Calculate the metric based on the given labels and probabilities.
        Args:
            labels (np.array): the ground truth labels.
            probs (np.array): the predicted probabilities.
        '''
        # process the predictions
        labels, probs = self.process_preds(labels, probs)
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return {self.get_metric_name: self.get_metric(labels, probs)}
            else:
                score = self.get_metric(labels, probs)
                return {k: v for k, v in zip(self.get_metric_name, score)}
        else:
            return {self.get_metric_name: self.get_metric(labels, probs)}

# -------------------------------------------------
# Original metric-calculation functions (no changes)
# -------------------------------------------------
def calculate_survival_metrics(logits: np.array,
                               labels: np.array,
                               label_dict: dict,
                               add_metrics: list=None) -> dict:
    # 默认只算 ['c_index']
    metrics = ['c_index'] + (add_metrics if add_metrics else [])
    results = {}
    for average in [None]:
        for metric in metrics:
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, logits))
    return results

def calculate_multilabel_metrics(probs: np.array, 
                                 labels: np.array, 
                                 label_dict, 
                                 add_metrics: list=None) -> dict: 
    metrics = ['auroc', 'auprc'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in ['micro', 'macro', None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results

def calculate_multiclass_or_binary_metrics(probs: np.array, 
                                           labels: np.array, 
                                           label_dict, 
                                           add_metrics: list=None) -> dict:
    metrics = ['bacc', 'acc', 'auroc', 'auprc', 'weighted_f1'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in ['macro', None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results

def calculate_regression_metrics(probs: np.array, 
                                 labels: np.array, 
                                 label_dict, 
                                 add_metrics: list=None) -> dict:
    metrics = ['mae', 'mse', 'rmse', 'pearson', 'spearman'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in [None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results

# -------------------------------------------------
# NEW: Text-generation metric function
# -------------------------------------------------
def calculate_text_generation_metrics(gts: dict, 
                                      res: dict, 
                                      label_dict: dict, 
                                      add_metrics: list=None) -> dict:
    """
    Calculate BLEU, METEOR, and ROUGE scores for text-generation tasks.
    
    Args:
        gts (dict): {sample_id: [reference_text_1, reference_text_2, ...], ...}
        res (dict): {sample_id: [generated_text], ...}
        label_dict (dict): Not strictly needed for text metrics, but included for consistency.
        add_metrics (list): If you want to add custom text metrics or filter. 
                            Currently unused but can be extended.

    Returns:
        dict: A dictionary of text metrics { 'BLEU_1': val, 'BLEU_2': val, 'BLEU_3': val,
                                             'BLEU_4': val, 'METEOR': val, 'ROUGE_L': val }
    """
    scores = compute_scores(gts, res)  # from the snippet you provided
    # If you have 'add_metrics' filters, you could apply them here. For now, returning all.
    return scores

# -------------------------------------------------
# Extend the main dispatch function
# -------------------------------------------------
def calculate_metrics_with_task_cfg(probs: np.array, 
                                    labels: np.array, 
                                    task_cfg: dict) -> dict:
    """
    Main entry to compute different metrics based on `task_cfg`.
    The 'probs' and 'labels' might be actual numeric arrays for classification/regression,
    or in the case of text generation tasks, you can pass dictionaries 
    that map IDs -> [text] for references/predictions.
    """
    task_setting = task_cfg.get('setting', 'multi_class')
    add_metrics = task_cfg.get('add_metrics', None)

    if task_setting == 'multi_label':
        return calculate_multilabel_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
    elif task_setting == 'regression':
        return calculate_regression_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
    elif task_setting == 'survival':
        return calculate_survival_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
    elif task_setting == 'text_generation':
        return calculate_text_generation_metrics(
            gts=labels, 
            res=probs, 
            label_dict=task_cfg['label_dict'], 
            add_metrics=add_metrics
        )
    else:
    
        return calculate_multiclass_or_binary_metrics(probs, labels, task_cfg['label_dict'], add_metrics)




"""
Contains visualization methods for classification tasks
"""

class RetrievalMixin:
    def retrieval_metrics(self, labels, preds, ks, saveto=None):
        """
        Compute retrieval metrics, both overall and per class.

        Args:
            labels (torch.Tensor): True labels for queries (shape: n).
            preds (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            ks (list): List of k values to compute metrics at.

        Returns:
            dict: Dictionary of retrieval metrics.
        """
        metrics = {'overall': {}}
        for k in ks:
            metrics['overall'][f'top{k}_acc'] = self.acc_at_k(preds, labels, k)
            metrics['overall'][f'mv@{k}_acc'] = self.mv_acc_at_k(preds, labels, k)
            metrics['overall'][f'mAP@{k}'] = self.map_at_k(preds, labels, k)

        # Compute metrics per class
        class_metrics = {}
        for cls in range(self.num_classes):
            cls_indices = (labels == cls)
            cls_labels = labels[cls_indices]
            cls_preds = preds[cls_indices]

            cls_metrics = {'support': len(cls_labels)}
            for k in ks:
                cls_metrics[f'top{k}_acc'] = self.acc_at_k(cls_preds, cls_labels, k)
                cls_metrics[f'mv@{k}_acc'] = self.mv_acc_at_k(cls_preds, cls_labels, k)
                cls_metrics[f'mAP@{k}'] = self.map_at_k(cls_preds, cls_labels, k)
            class_metrics[int(cls)] = cls_metrics

        metrics['per_class'] = class_metrics  # Add per-class metrics to the overall metrics

        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, 'w') as f:
                json.dump(metrics, f, indent=4)

        return metrics
    
    @staticmethod
    def acc_at_k(retrievals, y_queries, k):
        """
        Calculate accuracy at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'accuracy at k'.

        Returns:
            float: Accuracy at k.
        """
        topk_preds = retrievals[:, :k] # Shape: n x k
        return torch.any(topk_preds == y_queries[:, None], dim=1).float().mean().item() # if any of the topk matches, then it's correct

    @staticmethod
    def mv_acc_at_k(retrievals, y_queries, k):
        """
        Calculate majority vote accuracy at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'majority vote accuracy at k'.

        Returns:
            float: Majority vote accuracy at k.
        """
        topk_preds = retrievals[:, :k] # Shape: n x k
        all_uniques = [torch.unique(row, return_counts=True) for row in topk_preds] # Get majority vote for each row
        outcomes = []
        for label, (uniques, counts) in zip(y_queries, all_uniques):
            max_count = torch.max(counts)
            modes = uniques[counts == max_count] # If there are multiple modes, then it's a tie
            outcome = torch.isin(label, modes, assume_unique=True) # Check if label is in modes
            outcomes.append(outcome)
        return torch.tensor(outcomes).float().mean().item()
    
    @staticmethod
    def map_at_k(retrievals, y_queries, k):
        """
        Calculate mean average precision at k.

        Args:
            retrievals (torch.Tensor): Labels of top k retrievals from train set (shape: n x max(ks)).
            y_queries (torch.Tensor): True labels for queries (shape: n).
            k (int): The 'k' in 'mean average precision at k'.

        Returns:
            float: Mean average precision at k.
        """
        average_precisions = []
        for i, query_label in enumerate(y_queries):
            topk_labels = retrievals[i, :k]
            correct_count = 0
            precision_at_k = 0.0
            for j, label in enumerate(topk_labels):
                if label == query_label:
                    correct_count += 1
                    precision_at_k += correct_count / (j + 1)
            if correct_count > 0:
                average_precision = precision_at_k / k
            else:
                average_precision = 0.0
            average_precisions.append(average_precision)
        return np.mean(average_precisions)