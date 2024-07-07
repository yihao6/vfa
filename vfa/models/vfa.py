import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pdb
import numpy as np
import math

from vfa.models.base_model import BaseModel
from vfa.utils.utils import grid_sampler, identity_grid_like, keypoints_sampler, grid_upsample

class VFA(BaseModel):
    def __init__(self, params, device):
        super().__init__(params, device)

        self.dim = len(params['model']['in_shape'])
        self.encoder = Encoder(
                        dimension=self.dim,
                        in_channels=params['model']['in_channels'],
                        downsamples=params['model']['downsamples'],
                        start_channels=params['model']['start_channels'],
                        max_channels=64
        ).type(torch.float32)

        self.decoder = Decoder(
                        dimension=self.dim,
                        downsamples=params['model']['downsamples'],
                        matching_channels=params['model']['matching_channels'],
                        start_channels=params['model']['start_channels'],
                        max_channels=64,
                        skip=params['model']['skip'],
                        initialize=params['model']['initialize'],
                        int_steps=params['model']['int_steps'],
        ).type(torch.float32)

        self.to(device)
        self.decoder.R = self.decoder.R.to(device)

    def forward(self, sample):
        if 'f_input' in sample:
            F = self.encoder(sample['f_input'].to(self.device))
        else:
            F = self.encoder(sample['f_img'].to(self.device))

        if 'm_input' in sample:
            M = self.encoder(sample['m_input'].to(self.device))
        else:
            M = self.encoder(sample['m_img'].to(self.device))

        composed_grids = self.decoder(F, M)

        results = self.generate_results(composed_grids[-1], sample)
        results.update({'composed_grids': composed_grids,
                        'beta':self.decoder.beta.clone(),})

        if self.params['model']['affine']:
            affine_results = self.generate_affine_results(
                                                    composed_grids[-1],
                                                    sample,
            )
            results.update(affine_results)

        return results

    def calc_beta_loss(self, results, **args):
        '''for printing the value of Beta'''
        return results['beta']

class Encoder(nn.Module):
    '''VFA encoder network'''
    def __init__(self, dimension, in_channels, downsamples, start_channels, max_channels):
        super().__init__()
        self.dim = dimension
        self.downsamples = downsamples
        self.encoder_conv = nn.ModuleList()
        self.pooling =  nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.upsampling =  nn.ModuleList()

        Conv = getattr(nn, f'Conv{self.dim}d')
        Norm = getattr(nn, f'InstanceNorm{self.dim}d')
        Pooling = getattr(nn, f'AvgPool{self.dim}d')
        DoubleConv = globals()[f'DoubleConv{self.dim}d']
        Upsample = globals()[f'Upsample{self.dim}d']

        self.in_norm = Norm(in_channels, affine=True)
        self.in_conv = Conv(in_channels, start_channels, 3, padding=1)
        for i in range(self.downsamples):
            num_channels = start_channels * 2 ** i
            self.encoder_conv.append(DoubleConv(
                in_channels=min(max_channels, num_channels),
                mid_channels=min(max_channels, num_channels*2),
                out_channels=min(max_channels, num_channels*2)
            ))
            self.pooling.append(Pooling(2, stride=2))
            self.upsampling.append(Upsample(
                in_channels=min(max_channels, num_channels*4),
                out_channels=min(max_channels, num_channels*2),
            ))
            self.decoder_conv.append(DoubleConv(
                in_channels=min(max_channels, num_channels*2) * 2,
                mid_channels=min(max_channels, num_channels*2),
                out_channels=min(max_channels, num_channels*2),
            ))

        # bottleneck
        num_channels = start_channels * 2 ** self.downsamples
        self.bottleneck_conv = DoubleConv(
            in_channels=min(max_channels, num_channels),
            mid_channels=min(max_channels, num_channels*2),
            out_channels=min(max_channels, num_channels*2),
        )

    def forward(self, x):
        x = self.in_norm(x)
        x = nnf.leaky_relu(self.in_conv(x))
        feature_maps = []
        for i in range(self.downsamples):
            feature_maps.append(self.encoder_conv[i](x))
            x = self.pooling[i](feature_maps[-1])

        out_feature_maps = [self.bottleneck_conv(x)]

        for i in range(self.downsamples-1, -1, -1):
            x = self.upsampling[i](out_feature_maps[-1], feature_maps[i])
            out_feature_maps.append(self.decoder_conv[i](x))

        return out_feature_maps[::-1]

class Decoder(nn.Module):
    '''VFA decoder network'''
    def __init__(self, dimension, downsamples, matching_channels, start_channels, max_channels,
                 skip, initialize, int_steps):
        super().__init__()
        self.dim = dimension
        self.downsamples = downsamples
        self.project = nn.ModuleList()
        self.skip = skip
        self.int_steps = int_steps

        self.attention = Attention()
        Conv = getattr(nn, f'Conv{self.dim}d')
        self.beta = nn.Parameter(torch.tensor([float(initialize)]))

        self.similarity = 'inner_product'
        self.temperature = None

        for i in range(self.downsamples + 1):
            num_channels = start_channels * 2 ** i
            self.project.append(Conv(
                            min(max_channels, num_channels*2),
                            min(max_channels, matching_channels*2**i),
                            kernel_size=3,
                            padding=1,
            ))

        # precompute radial vector field r and R
        r = identity_grid_like(
                torch.zeros(1, 1, *[3 for _ in range(self.dim)]),
                normalize=True
        )
        self.R = self.to_token(r).squeeze().detach()

    def to_token(self, x):
        '''flatten the spatial dimensions and put the channel dimension to the end'''
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)
        return x

    def forward(self, F, M):
        composed_grids = []
        for i in range(self.downsamples, -1, -1):
            if i != self.downsamples:
                composed_grids[-1] = grid_upsample(composed_grids[-1])

            if i == 0 and self.skip:
                identity_grid = identity_grid_like(composed_grids[-1], normalize=False)
                composed_grids.append(grid_sampler(composed_grids[-1], identity_grid))
            else:
                # Q from the fixed image feature
                f = self.project[i](F[i])
                if self.similarity == 'cosine':
                    f = nnf.normalize(f, dim=1)
                permute_order = [0] + list(range(2, 2 + self.dim)) + [1]
                Q = f.permute(*permute_order).unsqueeze(-2)

                # K from the moving image feature maps
                m = grid_sampler(M[i], composed_grids[-1]) if len(composed_grids) != 0 else M[i]
                pad_size = [1 for _ in range(self.dim * 2)]
                m = nnf.pad(m, pad=tuple(pad_size), mode='replicate')
                m = self.project[i](m)
                if self.similarity == 'cosine':
                    m = nnf.normalize(m, dim=1)
                K = self.get_candidate_from_tensor(m, self.dim)

                # feature matching and location retrieval
                local_disp = self.attention(Q, K, self.R, self.temperature)
                permute_order = [0, -1] + list(range(1, 1 + self.dim))
                local_disp = local_disp.squeeze(-2).permute(*permute_order)
                identity_grid = identity_grid_like(local_disp, normalize=False)
                local_grid = local_disp * self.beta / 2**self.int_steps + identity_grid

                for _ in range(self.int_steps):
                    local_grid = grid_sampler(local_grid, local_grid)

                if i != self.downsamples:
                    composed_grids.append(grid_sampler(composed_grids[-1], local_grid))
                else:
                    composed_grids.append(local_grid.clone())

        return composed_grids

    def get_candidate_from_tensor(self, x, dim, kernel=3, stride=1):
        if dim == 3:
            '''from tensor with [Batch x Feature x Height x Weight x Depth],
                    extract patches [Batch x Feature x HxWxD x Patch],
                    and reshape to [Batch x HxWxS x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride).unfold(4, kernel, stride)
            patches = patches.flatten(start_dim=5)
            token = patches.permute(0, 2, 3, 4, 5, 1)
        elif dim == 2:
            '''From tensor with [Batch x Feature x Height x Weight],
                    extract patches [Batch x Feature x HxW x Patch],
                    and reshape to [Batch x HxW x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
            patches = patches.flatten(start_dim=4)
            token = patches.permute(0, 2, 3, 4, 1)

        return token

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = nnf.leaky_relu(self.norm1(self.conv1(x)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x)))
        return x

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = nnf.leaky_relu(self.norm1(self.conv1(x)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x)))
        return x

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, temperature):
        '''Expect input dimensions: [batch, *, feature]'''
        if temperature is None:
            temperature = key.size(-1) ** 0.5
        attention = torch.matmul(query, key.transpose(-1, -2)) / temperature
        attention = self.softmax(attention)
        x = torch.matmul(attention, value)
        return x

class Upsample3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x, feature_map):
        x = nnf.interpolate(
                x,
                size=None,
                scale_factor=2,
                mode='trilinear',
                align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map),dim=1)
        return x

class Upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, feature_map):
        x = nnf.interpolate(
                        x,
                        size=None,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map),dim=1)
        return x
