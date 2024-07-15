import json
import numpy as np
import torch
import nibabel as nib
import pathlib
import os
import torch.nn as nn
import pdb
import torch.nn.functional as nnf
import scipy.io
import logging

# Set up logging
def setup_logging():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to the terminal
        ]
    )

def load_data_configs(json_path):
    '''Load data configuration from json file'''
    if os.path.isfile(json_path) and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'loader': 'CL', 'shape':[1, 192, 224, 192]}

    # key labels is input of Dice loss
    if 'labels' not in data:
        data['labels'] = []

    return data

def update_params_json(path, params):
    '''Update parameter dictionary using .json file'''
    with open(path, 'r') as f:
        params_json = json.load(f)
    params.update(params_json)
    return params

def update_params_args(args, params):
    '''Update parameter dictionary using args'''
    for var in dir(args):
        if not var.startswith('_'):
            if var not in params:
                params[var] = getattr(args, var)
            elif getattr(args, var) is not None:
                params[var] = getattr(args, var)
    return params

def identity_grid_like(tensor, normalize, padding=0):
    '''return the identity grid for the input 2D or 3D tensor'''
    with torch.inference_mode():
        dims = tensor.shape[2:]
        if isinstance(padding, int):
            pads = [padding for j in range(len(dims))]
        else:
            pads = padding
        vectors = [torch.arange(start=0-pad, end=dim+pad) for (dim,pad) in zip(dims, pads)]

        try:
            grids = torch.meshgrid(vectors, indexing='ij')
        except TypeError:
            # compatible with old pytorch version
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)

        if normalize:
            in_dims = torch.tensor(dims).view(-1, len(dims), *[1 for x in range(len(dims))])
            grid = grid / (in_dims - 1) * 2 - 1

        grid = grid.to(tensor.device).repeat(tensor.shape[0],1,*[1 for j in range(len(dims))])
    return grid.clone()

def keypoints_sampler(source, keypoints, mode='bilinear', align_corners=True):
    '''sampmle the source tensor using the keypoints'''
    dims = torch.tensor(source.shape[2:]).to(keypoints.device)
    keypoints_norm = 2 * keypoints / (dims - 1) - 1
    if len(keypoints.shape) == 4:
        keypoints_norm = keypoints_norm[:,:,:,None,:]
    elif len(keypoints.shape) == 3:
        keypoints_norm = keypoints_norm[:,:,None,None,:]
    else:
        raise NotImplementedError
    out = nnf.grid_sample(
                            source,
                            keypoints_norm.flip(-1),
                            mode=mode,
                            align_corners=align_corners,
    )
    return out

def grid_upsample(grid, mode='bilinear', align_corners=True, scale_factor=2):
    '''upsample the grid by a factor of two'''

    if len(grid.shape[2:]) == 3 and mode == 'bilinear':
        mode = 'trilinear'

    in_dims = torch.tensor(grid.shape[2:]).to(grid.device)
    in_dims = in_dims.view(-1, len(in_dims), *[1 for x in range(len(in_dims))])
    out_dims = in_dims * scale_factor
    out = grid / (in_dims - 1) * (out_dims - 1)
    out = nnf.interpolate(
                            out,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
    )
    return out

def normalize_grid(grid):
    dims = torch.tensor(grid.shape[2:]).to(grid.device)
    dims = dims.view(-1, len(dims), *[1 for x in range(len(dims))])
    normed_grid = grid / (dims - 1) * 2 - 1 # convert to normalized space
    return normed_grid

def grid_sampler(source, grid, mode='bilinear', align_corners=True, normalize=True, padding_mode='border'):
    '''Grid sample with grid store in tensor format'''
    if normalize:
        normed_grid = normalize_grid(grid)
    else:
        normed_grid = grid

    if len(grid.shape[2:]) == 3:
        normed_grid = normed_grid.permute(0, 2, 3, 4, 1).flip(-1)
        warped = nnf.grid_sample(
                            source,
                            normed_grid,
                            align_corners=align_corners,
                            padding_mode=padding_mode,
                            mode=mode,
        )
    elif len(grid.shape[2:]) == 2:
        normed_grid = normed_grid.permute(0, 2, 3, 1).flip(-1)
        warped = nnf.grid_sample(
                            source,
                            normed_grid,
                            align_corners=align_corners,
                            padding_mode=padding_mode,
                            mode=mode,
        )
    return warped

def onehot(seg, labels):
    onehot = []
    for label in labels:
        onehot.append((seg==label))
    onehot = torch.cat(onehot, dim=1).type(seg.dtype)
    return onehot

def get_ants_affine_matrix(path):
    '''get the 4x4 numpy affine matrix (in physical space) from ANTs output'''
    raise NotImplementedError('To be released with VFA2.')

def calc_affine_grid(affine_matrix, fixed_affine, moving_affine, dims):
    raise NotImplementedError('To be released with VFA2.')

def save_results_to_xlsx(results, target_path):
    import pandas as pd

    data_df = pd.DataFrame({name: np.array(values) for name, values in results.items()})
    means = {name: np.mean(values) for name, values in results.items()}
    stds = {name: np.std(values) for name, values in results.items()}
    stats_df = pd.DataFrame({'mean': means, 'standard deviation': stds})

    # Save both dataframes to separate sheets of an Excel file
    with pd.ExcelWriter(target_path) as writer:
        data_df.to_excel(writer, sheet_name='raw data', index=False)
        stats_df.to_excel(writer, sheet_name='statistics', index=True)

def grid_lines(shape):
    DG = torch.zeros([1,1]+(list(shape)))
    DG[:,:,::16,:,:] = 1
    DG[:,:,:,::16,:] = 1
    # DG[:,:,:,:,::16] = 1 # remove axial plate
    # kernel = torch.ones((1,1,3,3,3)) / 3**3
    # DG = F.conv3d(DG, weight=kernel, stride=1, padding=1)
    # nib.save(nib.Nifti1Image(DG.squeeze().numpy(), np.eye(4)), './DG.nii.gz')
    return DG
