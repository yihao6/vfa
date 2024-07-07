import torch
import torch.nn as nn
import pdb
import math

from .base_model import BaseModel
from utils.utils import grid_sampler, identity_grid_like, keypoints_sampler, grid_upsample

class Im2grid(BaseModel):
    '''Liu, Yihao, et al. "Coordinate translator for learning deformable medical
       image registration." Multiscale Multimodal Medical Imaging: Third
       International Workshop, MMMI 2022'''
    def __init__(self, params, device):
        super().__init__(params, device)

        self.encoders = nn.ModuleDict()
        # initialize encoders depend on the number of input modalities
        # for modality in set(params['modalities']):
        #     self.encoders[modality] = Im2gridEncoder(
        #                                 in_channels=params['model']['in_channels'],
        #                                 downsamples=params['model']['downsamples'],
        #                                 start_channels=params['model']['start_channels'],
        #                                 max_channels=64,
        #     ).type(torch.float32)
        self.encoders = Im2gridEncoder(
                                    in_channels=params['model']['in_channels'],
                                    downsamples=params['model']['downsamples'],
                                    start_channels=params['model']['start_channels'],
                                    max_channels=64,
        ).type(torch.float32)

        self.decoder = Im2gridDecoder(
                                downsamples=params['model']['downsamples'],
                                matching_channels=params['model']['matching_channels'],
                                start_channels=params['model']['start_channels'],
                                max_channels=64,
                                skip=params['model']['skip']
        ).type(torch.float32)

        self.to(device)

    def forward(self, sample):
        # F = self.encoders[sample['f_mod'][0]](sample['f_img'].to(self.device))
        # M = self.encoders[sample['m_mod'][0]](sample['m_img'].to(self.device))
        F = self.encoders(sample['f_img'].to(self.device))
        M = self.encoders(sample['m_img'].to(self.device))

        composed_grids, local_grids, w_keypoints = self.decoder(F, M, None)
        results = self.generate_results(composed_grids[-1], sample)
        results.update({'composed_grids': composed_grids})

        return results

class Im2gridEncoder(nn.Module):
    def __init__(self, in_channels, downsamples, start_channels, max_channels):
        super().__init__()
        self.downsamples = downsamples
        self.encoder_conv = nn.ModuleList()
        self.pooling =  nn.ModuleList()

        self.in_conv = nn.Conv3d(in_channels, start_channels, 3, padding=1)
        for i in range(self.downsamples):
            num_channels = start_channels * 2 ** i
            self.encoder_conv.append(DoubleConv3d(
                    in_channels=min(max_channels, num_channels),
                    out_channels=min(max_channels, num_channels*2)
            ))
            self.pooling.append(nn.AvgPool3d(2, stride=2))

        # bottleneck
        num_channels = start_channels * 2 ** self.downsamples
        self.bottleneck_conv = nn.Conv3d(min(max_channels, num_channels),
                                         min(max_channels, num_channels), 3, padding=1)

    def forward(self, x):
        # CNN encoder
        x = nn.functional.leaky_relu(self.in_conv(x))
        out = []
        for i in range(self.downsamples):
            out.append(self.encoder_conv[i](x))
            x = self.pooling[i](out[-1])

        out.append(self.bottleneck_conv(x))
        return out

class Im2gridDecoder(nn.Module):
    def __init__(self, downsamples, matching_channels, start_channels, max_channels, skip):
        super().__init__()

        self.downsamples = downsamples
        self.positional_embedding = nn.ModuleList()
        self.coordinate_translator = CoordinateTranslator()
        self.skip = skip

        for i in range(self.downsamples):
            num_channels = start_channels * 2 ** i
            self.positional_embedding.append(PositionalEmbedding(
                    num_channels=min(max_channels, num_channels*2),
                    matching_channels=matching_channels
            ))

        # bottleneck
        num_channels = start_channels * 2 ** self.downsamples
        self.positional_embedding.append(PositionalEmbedding(
            num_channels=min(max_channels, num_channels),
            matching_channels=matching_channels
        ))

    def forward(self, F, M, f_keypoints):
        # Grid decoder
        composed_grids = [identity_grid_like(M[-1], normalize=True)]
        local_grids = []

        for i in range(self.downsamples,-1,-1):
            # token from the moving image feature maps
            M[i] = grid_sampler(M[i], composed_grids[-1], normalize=False)
            if i in self.skip:
                pass
            else:
                m = nn.functional.pad(M[i], pad=(1,1,1,1,1,1)) # zero padding before extract patches
                grid = self.get_positional_embedding(M[i], padding=1)
                m = self.positional_embedding[i](m, grid)
                token_m = self.tensor_to_patch_token(m, 3, 1)

            # token from the fixed image feature
            if i in self.skip:
                pass
            else:
                grid = self.get_positional_embedding(F[i], padding=0)
                f = self.positional_embedding[i](F[i], grid)
                token_f = self.combine_spatial_dim(f).unsqueeze(-2)

            # token from the moving image grid
            if i in self.skip:
                pass
            else:
                grid = identity_grid_like(M[i], normalize=True, padding=1)
                token_g = self.tensor_to_patch_token(grid, 3, 1)

            # coordinate translator
            if i in self.skip:
                tensor_grid = identity_grid_like(M[i], normalize=True)
            else:
                token = self.coordinate_translator(token_f, token_m, token_g).squeeze(-2)
                tensor_grid = self.separate_spatial_dim(token, dims=F[i].shape[2:])

            composed_grids.append(grid_sampler(composed_grids[-1], tensor_grid, normalize=False))
            local_grids.append(tensor_grid)
            if i != 0:
                for j in range(len(composed_grids)):
                    composed_grids[j] = nn.functional.interpolate(composed_grids[j], size=None,
                        scale_factor=2, mode='trilinear', align_corners=True)
                for j in range(len(local_grids)):
                    local_grids[j] = nn.functional.interpolate(local_grids[j], size=None,
                        scale_factor=2, mode='trilinear', align_corners=True)

        # if f_keypoints is provided, sample the dense grid to get
        # the transformation at the keypoint location
        if f_keypoints is not None:
            w_keypoints = keypoints_sampler(composed_grids[-1], f_keypoints)
            w_keypoints = w_keypoints.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        else:
            w_keypoints = None

        # convert grid to pixel spacing
        dims = composed_grids[-1].shape[2:]
        dims = torch.tensor(dims).view(-1, len(dims), *[1 for x in range(len(dims))])
        dims = dims.to(composed_grids[-1].device)
        for i in range(len(composed_grids)):
            composed_grids[i] = (composed_grids[i] + 1) / 2 * (dims - 1)

        for i in range(len(local_grids)):
            local_grids[i] = (local_grids[i] + 1) / 2 * (dims - 1)

        return composed_grids, local_grids, w_keypoints

    def get_positional_embedding(self, tensor, padding=0):
        if isinstance(padding, int):
            pads = [padding, padding, padding]
        else:
            pads = padding
        dims = tensor.shape[2:]
        vectors = [torch.arange(start=0-pad, end=dim+pad) for (dim,pad) in zip(dims, pads)]

        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)

        grid = 2 * (grid/(max(dims)-1) - 0.5)
        # map the each coordinate dimension to a unit vector
        grid = torch.cat((torch.cos(grid*math.pi/2.0), torch.sin(grid*math.pi/2.0)), dim=1)
        return grid.to(tensor.device).repeat(tensor.shape[0],1,1,1,1).detach()

    def tensor_to_patch_token(self, x, kernel=3, stride=1):
        '''from tensor, extract patches and reshape'''
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride).unfold(4, kernel, stride)
        patches = patches.flatten(start_dim=5)
        patches = patches.flatten(start_dim=2, end_dim=-2)
        token = patches.permute(0, 2, 3, 1)
        return token

    def patch_token_to_tensor(self, token, shape):
        token = token.permute(0, 3, 1, 2)
        token = token.view(list(token.shape[:2]) + list(shape[1]) + [2,2,2])
        token = token.permute(0, 1, 2, 5, 3, 6, 4, 7)
        tensor = token.reshape(list(token.shape[:2]) + list(shape[0]))
        return tensor

    def separate_spatial_dim(self, x, dims):
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1) 
            x = x.view(x.shape[0], x.shape[1], *tuple(dims))
        return x

    def combine_spatial_dim(self, x):
        if len(x.shape) == 4:
            x = x.flatten(start_dim=2)
            x = x.transpose(-1, -2)
        if len(x.shape) == 5:
            x = x.flatten(start_dim=2)
            x = x.transpose(-1, -2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, matching_channels):
        super().__init__()
        assert matching_channels >= 6
        self.tensor_proj = nn.Conv3d(num_channels, matching_channels, 1)
        self.tensor_proj.weight.data.zero_()
        self.tensor_proj.bias.data.zero_()

        self.grid_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, grid):
        x = self.tensor_proj(x)
        x[:,:6,...] += grid * self.grid_scale
        return x

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.norm1(self.conv1(x)))
        x = nn.functional.leaky_relu(self.norm2(self.conv2(x)))
        return x

class CoordinateTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = self.softmax(attention)
        out = torch.matmul(attention, value)
        return out
