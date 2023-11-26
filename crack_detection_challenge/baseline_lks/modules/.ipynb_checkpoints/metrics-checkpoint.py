"""Metric 함수 정의
"""

import torch
import numpy as np

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """
    
    if metric_function_str == 'iou':
        return IOU
        
def IOU(outputs: torch.Tensor, labels: torch.Tensor, eps=1e-6):
    batch_size = outputs.size()[0]
    
    intersection = ((outputs.int() == 1) & (labels.int() == 1) & (outputs.int() == labels.int())).float()
    intersection = intersection.view(batch_size, -1).sum(1)

    union = ((outputs.int() == 1) | (labels.int() == 1)).float()
    union = union.view(batch_size, -1).sum(1)

    iou = (intersection + eps) / (union + eps)

    return iou.mean()


