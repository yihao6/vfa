import torch.nn as nn
import torch
import pdb

class MSE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target, pred_mask=None, target_mask=None):
        assert target.shape == pred.shape, 'MSE loss requires input shape to be the same'

        if pred_mask is not None:
            pred = pred * pred_mask * target_mask
            target = target * pred_mask * target_mask
            return torch.sum((target - pred) ** 2) / torch.sum(pred_mask * target_mask)
        else:
            return torch.mean((target - pred) ** 2)
