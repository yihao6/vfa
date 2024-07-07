import torch.nn as nn
import torch
import pdb
import torch.nn.functional as nnf
import numpy as np
import math

class SingleScaleNCC(nn.Module):
    def __init__(self, window_size, **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            self.window_size = [window_size]
        else:
            self.window_size = window_size

    def forward(self, pred, target):
        """ LNCC loss
            modified based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        """
        Ii = target
        Ji = pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = self.window_size * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(pred.device) / np.prod(win)

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, sum_filt, padding=padding, stride=stride)
        mu2 = conv_fn(Ji, sum_filt, padding=padding, stride=stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, sum_filt, padding=padding, stride=stride) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, sum_filt, padding=padding, stride=stride) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, sum_filt, padding=padding, stride=stride) - mu1_mu2

        eps = torch.finfo(sigma12.dtype).eps
        cc = (sigma12 * sigma12) / torch.clamp(sigma1_sq * sigma2_sq, min=eps)
        return - torch.mean(cc)


class NCC(torch.nn.Module):
    """
    Multi-scale NCC from C2FViT: https://github.com/cwmok/C2FViT
    """
    def  __init__(self, **kwargs):
        super().__init__()
        self.num_scales = kwargs['scale'] if 'scale' in kwargs else 1
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs else 3
        self.half_resolution = kwargs['half_resolution'] if 'half_resolution' in kwargs else 0

        self.similarity_metric = []
        for i in range(self.num_scales):
            self.similarity_metric.append(
                        SingleScaleNCC(kwargs['window_size']-(i*2))
            )

    def forward(self, I, J):
        dim = I.dim() - 2
        if self.half_resolution:
            kwargs = {'scale_factor':0.5, 'align_corners':True}
            if dim == 2:
                I = nnf.interpolate(I, mode='bilinear', **kwargs)
                J = nnf.interpolate(J, mode='bilinear', **kwargs)
            elif dim == 3:
                I = nnf.interpolate(I, mode='trilinear', **kwargs)
                J = nnf.interpolate(J, mode='trilinear', **kwargs)

        if dim == 2:
            pooling_fn = nnf.avg_pool2d
        elif dim == 3:
            pooling_fn = nnf.avg_pool3d

        total_NCC = []
        for i in range(self.num_scales):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC / self.num_scales)

            I = pooling_fn(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = pooling_fn(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)
