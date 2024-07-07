import torch.nn as nn
import torch
import pdb

class Grad(nn.Module):
    """Gradient loss using l2 norm"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred):
        dim = pred.dim() - 2  # Determine the dimension from the input tensor
        if dim == 2:
            return self.compute_2d_loss(pred)
        elif dim == 3:
            return self.compute_3d_loss(pred)
        else:
            raise ValueError(f"Invalid input shape. Only 2D and 3D tensors are supported.")

    def compute_2d_loss(self, pred):
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]

        dx = dx * dx
        dy = dy * dy

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad

    def compute_3d_loss(self, pred):
        dx = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dz = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

        dx = dx * dx
        dy = dy * dy
        dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad
