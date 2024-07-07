import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import pdb

class CorrRatio(torch.nn.Module):
    """
    Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.sigma_ratio = kwargs['sigma_ratio'] if 'sigma_ratio' in kwargs else 1 / 2.355
        self.f_num_bins = kwargs['f_num_bins'] if 'f_num_bins' in kwargs else 32
        self.m_num_bins = kwargs['m_num_bins'] if 'm_num_bins' in kwargs else 32

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))#torch.exp(-0.5 * (diff ** 2) / (sigma ** 2))

    def correlation_ratio(self, pred, target):
        self.f_max_val = torch.quantile(target, 0.99).item()
        self.f_min_val = torch.quantile(target, 0.01).item()
        self.m_max_val = torch.quantile(pred, 0.99).item()
        self.m_min_val = torch.quantile(pred, 0.01).item()

        f_bin_size = (self.f_max_val - self.f_min_val) / self.f_num_bins
        self.f_vbc = torch.linspace(
                            self.f_min_val + f_bin_size / 2,
                            self.f_max_val - f_bin_size / 2,
                            self.f_num_bins
        ).view(1, 1, self.f_num_bins, 1)
        f_sigma = torch.mean(torch.diff(self.f_vbc.squeeze())).item() * self.sigma_ratio
        self.f_preterm = 1 / (2 * f_sigma ** 2)

        # 
        y = torch.clamp(target, self.f_min_val, self.f_max_val)
        x = torch.clamp(pred, self.m_min_val, self.m_max_val)

        y_flat = y.view(y.shape[0], y.shape[1], -1)
        x_flat = x.view(x.shape[0], x.shape[1], -1)

        eps_div = torch.finfo(y_flat.dtype).eps  # Epsilon for division
        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B, C, 1, H*W*D]
        diff = y_expanded - self.f_vbc.to(y_expanded.device)  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.f_preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True) + eps_div)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B, C, 1, H*W*D]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True) # [B, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(
                    bin_counts * (mean_intensities - total_mean) ** 2,
                    dim=2
        ) / (torch.sum(bin_counts, dim=2) + eps_div)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + eps_div)

        return eta_square.mean()

    def forward(self, y_true, y_pred):
        symmetric_cr = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -symmetric_cr / 2

class LocalCorrRatio(torch.nn.Module):
    """
    Localized Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.num_bins = kwargs['bins'] if 'bins' in kwargs else 32
        self.sigma_ratio = kwargs['sigma_ratio'] if 'sigma_ratio' in kwargs else 1
        self.win = kwargs['win'] if 'win' in kwargs else 9

        bin_centers = np.linspace(0, 1, num=self.num_bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, self.num_bins), requires_grad=False).cuda().view(1, 1, self.num_bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * self.sigma_ratio
        # print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape

        h_r = -H % self.win
        w_r = -W % self.win
        d_r = -D % self.win
        padding = (d_r // 2, d_r - d_r // 2, w_r // 2, w_r - w_r // 2, h_r // 2, h_r - h_r // 2, 0, 0, 0, 0)
        X = nnf.pad(X, padding, "constant", 0)
        Y = nnf.pad(Y, padding, "constant", 0)

        B, C, H, W, D = Y.shape
        num_patch = (H // self.win) * (W // self.win) * (D // self.win)
        x_patch = torch.reshape(X, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        x_flat = x_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B*num_patch, C, self.win ** 3)

        y_patch = torch.reshape(Y, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        y_flat = y_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B * num_patch, C, self.win ** 3)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B*num_patch, C, 1, win**3]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B*num_patch, C, 1, win**3]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B*num_patch, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True)  # [B*num_patch, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / torch.sum(
            bin_counts, dim=2)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean() / 3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true) #make it symmetric

        shift_size = self.win//2
        y_true = torch.roll(y_true, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))
        y_pred = torch.roll(y_pred, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))

        CR_shifted = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/4 - CR_shifted/4
