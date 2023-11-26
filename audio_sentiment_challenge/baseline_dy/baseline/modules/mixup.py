import torch
import numpy as np

def MixUp(input_, target, config):
    if config.alpha > 0:
        lambda_ = np.random.beta(config.alpha, config.alpha)
    else:
        lambda_ = 1
 
    batch_size = input_.size(0)
    index = torch.randperm(batch_size)
    
    mixed_input = lambda_ * input_ + (1 - lambda_) * input_[index, :]
    labels_a, labels_b = target, target[index]
 
    return mixed_input, labels_a, labels_b, lambda_

def MixUpLoss(criterion, pred, labels_a, labels_b, lambda_):
    return lambda_ * criterion(pred, labels_a) + (1 - lambda_) * criterion(pred, labels_b)