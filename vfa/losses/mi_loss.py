import torch.nn as nn
import torch
import math
import pdb

class MI(nn.Module):
    """ Mutual Information loss
        modified based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.sigma_ratio = 1 / 2.355 # make the FWHM of Gaussian equals bin size
        self.f_num_bins = kwargs['f_num_bins'] if 'f_num_bins' in kwargs else 32
        self.m_num_bins = kwargs['m_num_bins'] if 'm_num_bins' in kwargs else 32

    def forward(self, pred, target):
        '''setup'''
        self.f_max_val = torch.quantile(target, 0.99).item()
        self.f_min_val = torch.quantile(target, 0.01).item()
        self.m_max_val = torch.quantile(pred, 0.99).item()
        self.m_min_val = torch.quantile(pred, 0.01).item()

        f_bin_size = (self.f_max_val - self.f_min_val) / self.f_num_bins
        self.f_vbc = torch.linspace(
                            self.f_min_val + f_bin_size / 2,
                            self.f_max_val - f_bin_size / 2,
                            self.f_num_bins
        )
        f_sigma = torch.mean(torch.diff(self.f_vbc)).item() * self.sigma_ratio
        try:
            self.f_preterm = 1 / (2 * f_sigma ** 2)
        except:
            pdb.set_trace()

        m_bin_size = (self.m_max_val - self.m_min_val) / self.m_num_bins
        self.m_vbc = torch.linspace(
                                self.m_min_val + m_bin_size / 2,
                                self.m_max_val - m_bin_size / 2,
                                self.m_num_bins
        )
        m_sigma = torch.mean(torch.diff(self.m_vbc)).item() * self.sigma_ratio
        try:
            self.m_preterm = 1 / (2 * m_sigma ** 2)
        except:
            pdb.set_trace()

        #
        a = torch.clamp(target, self.f_min_val, self.f_max_val)
        b = torch.clamp(pred, self.m_min_val, self.m_max_val)

        a = a.view(a.shape[0], -1)
        b = b.view(b.shape[0], -1)

        a = torch.unsqueeze(a, 2)
        b = torch.unsqueeze(b, 2)
        num_voxels = b.shape[1]

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.f_preterm * torch.square(a - self.f_vbc[None,None,...].to(pred.device)))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True) # normalize by total area of window

        I_b = torch.exp(- self.m_preterm * torch.square(b - self.m_vbc[None,None,...].to(pred.device)))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)
        # compute probabilities
        scale = math.sqrt(num_voxels)
        pab = torch.bmm(I_a.permute(0, 2, 1) / scale, I_b / scale)
        papb = torch.bmm(pa.permute(0, 2, 1), pb)

        eps_div = torch.finfo(pab.dtype).eps  # Epsilon for division
        eps_log = torch.finfo(pab.dtype).tiny  # Epsilon for logarithm
        div_term = pab / (papb + eps_div)
        log_term = torch.log(div_term + eps_log)
        mi = torch.sum(torch.sum(pab * log_term, dim=1), dim=1)

        return -mi.mean()  # average across batch
