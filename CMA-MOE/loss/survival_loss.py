import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def neg_log_partial_likelihood(preds, label):
    preds = preds.squeeze()
    times = label[:,0]
    events = label[:,1]
    risk_order = torch.argsort(times, descending=True)
    preds, times, events = preds[risk_order], times[risk_order], events[risk_order]
    
    log_risk = torch.log(torch.cumsum(torch.exp(preds), dim=0))
    uncensored_likelihood = preds - log_risk
    loss = -torch.sum(uncensored_likelihood * events) / events.sum()
    return loss

