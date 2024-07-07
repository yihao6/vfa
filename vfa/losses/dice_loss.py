import torch.nn as nn
import torch
import pdb

class Dice(nn.Module):
    """From Voxelmorph, N-D dice for segmentation."""
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target):
        return (1 - 1.0 * self.calc_dice(pred, target))

    def calc_dice(self, pred, target):
        eps = torch.finfo(pred.dtype).eps
        ndims = len(list(pred.size())) - 2
        spatial_dims = list(range(2, ndims + 2))
        top = 2 * (target * pred).sum(dim=spatial_dims)
        bottom = torch.clamp((target + pred).sum(dim=spatial_dims), min=eps)
        dice = top / bottom
        return dice
