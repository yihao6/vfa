import torch.nn as nn
import torch
import pdb
import logging

from vfa.utils.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class Bending(nn.Module):
    '''The smoothness term defined in "Nonrigid Registration Using Free-Form
       Deformations: Application to Breast MR Images"'''
    def __init__(self, **kwargs):
        super().__init__()
        logger.warning('Current implementation of Bending energy loss assumes 1mm isotropic resolution.')

    def forward(self, pred, voxel_dims):
        dim = pred.dim() - 2  # Determine the dimension from the input tensor
        if dim == 2:
            return self.compute_2d_loss(pred, voxel_dims[:, :2])
        elif dim == 3:
            return self.compute_3d_loss(pred, voxel_dims)
        else:
            raise ValueError(f"Invalid input shape. Only 2D and 3D tensors are supported.")

    def compute_2d_loss(self, pred, voxel_dims):
        px = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) / voxel_dims[:, 0].view(-1, 1, 1, 1)
        py = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) / voxel_dims[:, 1].view(-1, 1, 1, 1)

        pxx = (px[:, :, 1:, :] - px[:, :, :-1, :]) / voxel_dims[:, 0].view(-1, 1, 1, 1)
        pyy = (py[:, :, :, 1:] - py[:, :, :, :-1]) / voxel_dims[:, 1].view(-1, 1, 1, 1)
        pxy = (px[:, :, :, 1:] - px[:, :, :, :-1]) / voxel_dims[:, 1].view(-1, 1, 1, 1)

        bending = torch.mean(pxx ** 2) + torch.mean(pyy ** 2) + torch.mean(2 * pxy ** 2)

        return bending

    def compute_3d_loss(self, pred, voxel_dims):
        px = (pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]) / voxel_dims[:, 0].view(-1, 1, 1, 1, 1)
        py = (pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]) / voxel_dims[:, 1].view(-1, 1, 1, 1, 1)
        pz = (pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) / voxel_dims[:, 2].view(-1, 1, 1, 1, 1)

        pxx = (px[:, :, 1:, :, :] - px[:, :, :-1, :, :]) / voxel_dims[:, 0].view(-1, 1, 1, 1, 1)
        pxy = (px[:, :, :, 1:, :] - px[:, :, :, :-1, :]) / voxel_dims[:, 1].view(-1, 1, 1, 1, 1)
        pxz = (px[:, :, :, :, 1:] - px[:, :, :, :, :-1]) / voxel_dims[:, 2].view(-1, 1, 1, 1, 1)
        pyy = (py[:, :, :, 1:, :] - py[:, :, :, :-1, :]) / voxel_dims[:, 1].view(-1, 1, 1, 1, 1)
        pyz = (py[:, :, :, :, 1:] - py[:, :, :, :, :-1]) / voxel_dims[:, 2].view(-1, 1, 1, 1, 1)
        pzz = (pz[:, :, :, :, 1:] - pz[:, :, :, :, :-1]) / voxel_dims[:, 2].view(-1, 1, 1, 1, 1)

        bending = (torch.mean(pxx ** 2) + torch.mean(2 * pxy ** 2) +
                   torch.mean(2 * pxz ** 2) + torch.mean(pyy ** 2) +
                   torch.mean(2 * pyz ** 2) + torch.mean(pzz ** 2)
        )
        return bending
