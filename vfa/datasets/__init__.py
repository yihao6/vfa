import pdb
import os
import importlib.util
import logging
from vfa.utils.utils import grid_sampler, grid_lines, identity_grid_like, setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Directory containing dataset files
datasets_directory = os.path.dirname(__file__)

def load_dataset(dataset_name):
    """
    Load the dataset class of the specified dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        dataset_class: Class of the dataset.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        AttributeError: If the dataset class is not found.
    """

    dataset_file_path = os.path.join(datasets_directory, f"{dataset_name.lower()}_dataset.py")
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Dataset file '{dataset_name.lower()}_dataset.py' not found in '{datasets_directory}'.")

    dataset_module_name = f"datasets.{dataset_name.lower()}_dataset"
    dataset_module_spec = importlib.util.spec_from_file_location(
                                    dataset_module_name,
                                    dataset_file_path
    )
    dataset_module = importlib.util.module_from_spec(dataset_module_spec)
    dataset_module_spec.loader.exec_module(dataset_module)

    dataset_class_name = dataset_name + 'Dataset'
    dataset_class = getattr(dataset_module, dataset_class_name, None)

    if dataset_class is None:
        raise AttributeError(f"Dataset class '{dataset_class_name}' not found in module '{dataset_module_name}'.")

    return dataset_class

from nibabel.processing import resample_from_to, resample_to_output
from nibabel.orientations import axcodes2ornt, ornt_transform
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torchvision import transforms
from random import randint
import random
import numpy as np
import torch
from pathlib import Path
import nibabel as nib

class BaseDataset(Dataset, ABC):
    def __init__(self, configs, params):
        super().__init__()
        if 'transform' not in configs:
            configs['transform'] = [
                {"class_name":"Reorient", "orientation":'RAS'},
                {"class_name":"Resample", "target_res":[1.0, 1.0, 1.0]},
                {"class_name":"Nifti2Array"},
                {"class_name":"AdjustShape", "target_shape":[192, 224, 192]},
                {"class_name":"DatatypeConversion"},
                {"class_name":"ToTensor"},
            ]
            logger.info('No transform specified in data_config file. Use default.')

        self.configs = configs
        self.params = params

        self.transforms = self.init_transforms(configs['transform'])
        self.configs['dimension'] = len(self.configs['shape'][1:])

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def reverse_shape_adjustment(self, arr, input_shape):
        # Initialize the result with the input array
        result = arr

        for i in range(len(input_shape)):
            original_dim = input_shape[i]
            current_dim = arr.shape[i]

            if original_dim > current_dim:
                # The original image was cropped, so we pad it back.
                # Calculate padding to add on both sides
                padding_before = (original_dim - current_dim) // 2
                padding_after = original_dim - current_dim - padding_before
                padding = [(0, 0)] * i + [(padding_before, padding_after)] + [(0, 0)] * (len(arr.shape) - i - 1)
                result = np.pad(result, padding, mode='constant', constant_values=0)
            elif original_dim < current_dim:
                # The original image was padded, so we remove the padding.
                # Calculate the slicing indices to remove padding
                start = (current_dim - original_dim) // 2
                end = start + original_dim
                result = result.take(indices=range(start, end), axis=i)

        return result

    def save_img(self, img, affine, target_path, shape=None, reference_obj=None):
        if self.configs['dimension'] == 2:
            raise NotImplementedError('2D data is not supported in the current version.')
            out = img.permute(0, 2, 3, 1).cpu().squeeze().numpy()
            cv2.imwrite(str(target_path), (255 * out).astype('uint8'))
        else:
            out = img.permute(0, 2, 3, 4, 1).cpu().squeeze().numpy()

            # undo shape adjustment
            if shape is not None:
                out = self.reverse_shape_adjustment(out, shape)

            if reference_obj is None:
                img_obj = nib.Nifti1Image(out, affine.squeeze().numpy())
                nib.save(img_obj, str(target_path))
            else:
                img_obj = nib.Nifti1Image(out, affine.squeeze().numpy())
                resampled_img = resample_from_to(img_obj, reference_obj)
                nib.save(resampled_img, str(target_path))

    def load_img_obj(self, img_path):
        if not os.path.exists(str(img_path)):
            raise FileNotFoundError(f"Cannot find image {img_path}.")

        if self.configs['dimension'] == 2:
            raise NotImplementedError('2D data is not supported in the current version.')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            affine = np.eye(4)
            affine[0, 0] = affine[1, 1] = -1
            return nib.Nifti1Image(img, affine)
        else:
            # always convert voxel array to standard coordinate
            img_obj = nib.load(img_path)
            return img_obj

    def init_transforms(self, config):
        '''initialize preprocessing transforms based on config'''
        transform_list = []
        for item in config:
            class_name = item.pop('class_name')
            transform_class = getattr(type(self), class_name, None)
            if transform_class:
                transform = transform_class() if not item else transform_class(**item)
                transform_list.append(transform)
            else:
                raise ValueError(f"Transform '{name}' not found in class '{type(self).__name__}'")
        return transforms.Compose(transform_list)

    def export(self, results, sample):
        if self.params['save_results'] == 0:
            return None

        # mkdir
        mkdir_kwargs = {'parents': True, 'exist_ok': True}
        prefix = sample['prefix'][0]
        Path(prefix).parents[0].mkdir(**mkdir_kwargs)

        f_ref_obj = self.load_img_obj(sample['f_img_path'][0])
        m_ref_obj = self.load_img_obj(sample['m_img_path'][0])
        f_shape = tuple(x.item() for x in sample['f_shape']) if 'f_shape' in sample else None
        m_shape = tuple(x.item() for x in sample['m_shape']) if 'm_shape' in sample else None

        # compose the external affine grid if provided
        if 'ext_affine_grid' in sample:
            sample['ext_affine_grid'] = sample['ext_affine_grid'].to(results['grid'].device)
            results['grid'] = grid_sampler(sample['ext_affine_grid'], results['grid'])

        if self.params['save_results'] == 2:
            # fixed images
            self.save_img(
                            sample['f_img'],
                            sample['f_affine'],
                            prefix + '_f_img.nii.gz',
                            shape=f_shape,
                            reference_obj = f_ref_obj
            )
            # moving image
            self.save_img(
                            sample['m_img'],
                            sample['m_affine'],
                            prefix + '_m_img.nii.gz',
                            shape=m_shape,
                            reference_obj = m_ref_obj
            )
            # warped mask
            self.save_img(
                            results['w_mask'],
                            sample['f_affine'],
                            prefix + '_w_mask.nii.gz',
                            shape=f_shape,
                            reference_obj = f_ref_obj
            )

            if 'f_seg' in sample:
                self.save_img(
                                sample['f_seg'],
                                sample['f_affine'],
                                prefix + '_f_seg.nii.gz',
                                shape=f_shape,
                                reference_obj = f_ref_obj
                )

            if 'm_seg' in sample:
                self.save_img(
                                sample['m_seg'],
                                sample['m_affine'],
                                prefix + '_m_seg.nii.gz',
                                shape=m_shape,
                                reference_obj = m_ref_obj
                )

            # grid line representation
            grid_lines_rep = grid_lines(sample['f_img'].shape[2:])
            grid_lines_rep = grid_sampler(
                            grid_lines_rep.to(results['grid'].device),
                            results['grid'],
            )
            self.save_img(
                            grid_lines_rep,
                            torch.from_numpy(np.eye(4)),
                            prefix + '_grid_lines.nii.gz',
            )

            # displacement magnitude
            disp = results['grid'] - identity_grid_like(results['grid'], normalize=False)
            magnitude = (disp ** 2).sum(dim=1).sqrt().unsqueeze(1)
            self.save_img(
                            magnitude,
                            torch.from_numpy(np.eye(4)),
                            prefix + '_disp_magnitude.nii.gz',
            )
            if self.params['model']['affine']:
                nonlinear_grid = grid_sampler(
                            results['inv_affine_grid'],
                            results['grid'],
                )
                disp = nonlinear_grid - identity_grid_like(results['grid'], normalize=False)
                magnitude = (disp ** 2).sum(dim=1).sqrt().unsqueeze(1)
                self.save_img(
                                magnitude,
                                torch.from_numpy(np.eye(4)),
                                prefix + '_disp_magnitude_nonlinear.nii.gz',
                )

                # grid line representation
                grid_lines_rep = grid_lines(sample['f_img'].shape[2:])
                grid_lines_rep = grid_sampler(
                                grid_lines_rep.to(nonlinear_grid.device),
                                nonlinear_grid,
                )
                self.save_img(
                                grid_lines_rep,
                                torch.from_numpy(np.eye(4)),
                                prefix + '_grid_lines_nonlinear.nii.gz',
                )

        # warped image
        self.save_img(
                results['w_img'] * results['w_mask'],
                sample['f_affine'],
                prefix + '_w_img.nii.gz',
                shape=f_shape,
                reference_obj = f_ref_obj
        )

        # mask intersect
        self.save_img(
                    results['mask'],
                    sample['f_affine'],
                    prefix + '_mask.nii.gz',
                    shape=f_shape,
                    reference_obj = f_ref_obj
            )

        # grid
        self.save_img(
                        results['grid'],
                        torch.from_numpy(np.eye(4)),
                        prefix + '_grid.nii.gz',
        )


        if 'm_seg' in sample:
            if 'ext_affine_grid' in sample:
                sample['ext_affine_grid'] = sample['ext_affine_grid'].to(results['grid'].device)
                results['grid'] = grid_sampler(sample['ext_affine_grid'], results['grid'])

            results['w_seg'] = grid_sampler(
                                    sample['m_seg'].to(results['grid'].device),
                                    results['grid'],
                                    mode='nearest',
            )

            self.save_img(
                    results['w_seg'] * results['w_mask'],
                    sample['f_affine'],
                    prefix + '_w_seg.nii.gz',
                    shape=f_shape,
                    reference_obj = f_ref_obj
            )

    # transforms
    class Reorient(object):
        def __init__(self, orientation='RAS'):
            self.standard_orientation = axcodes2ornt(orientation)

        def reorient(self, img_obj):
            orientation = axcodes2ornt(nib.aff2axcodes(img_obj.affine))
            transform_to_standard = ornt_transform(orientation, self.standard_orientation)
            img_obj_reoriented = img_obj.as_reoriented(transform_to_standard)
            return img_obj_reoriented

        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'f_seg', 'f_mask', 'f_input', 'm_img', 'm_seg', 'm_mask', 'm_input']:
                    sample[key] = self.reorient(sample[key])

            return sample

    class Resample(object):
        def __init__(self, target_res=(1.0, 1.0, 1.0)):
            self.target_res = target_res

        def resample(self, img_obj, order):
            source_res = tuple(round(num, 1) for num in img_obj.header.get_zooms())
            if source_res != self.target_res:
                return resample_to_output(img_obj, self.target_res, order)
            else:
                return img_obj

        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'f_input', 'm_img', 'm_input']:
                    sample[key] = self.resample(sample[key], order=1)

                if key in ['f_mask', 'f_seg', 'm_mask', 'm_seg']:
                    sample[key] = self.resample(sample[key], order=0)

            return sample

    class Nifti2Array(object):
        def __call__(self, sample):
            sample['f_img'], sample['f_affine'] = sample['f_img'].get_fdata(), sample['f_img'].affine
            sample['m_img'], sample['m_affine'] = sample['m_img'].get_fdata(), sample['m_img'].affine

            for key in sample:
                if key in ['f_seg', 'f_mask', 'f_input', 'm_seg', 'm_mask', 'm_input']:
                    sample[key] = sample[key].get_fdata()

            return sample

    class AdjustShape(object):
        def __init__(self, target_shape=(192, 224, 192)):
            self.target_shape = target_shape

            # Check if target shape is valid (multiples of 16)
            if any(dim % 16 != 0 for dim in self.target_shape):
                raise ValueError("Target shape dimensions must be multiples of 16")

        def adjust_to_target_shape(self, arr):
            # Pad or crop each dimension to target
            adjusted = arr
            for i in range(len(arr.shape)):
                if arr.shape[i] < self.target_shape[i]:
                    # Padding
                    padding_before = (self.target_shape[i] - arr.shape[i]) // 2
                    padding_after = self.target_shape[i] - arr.shape[i] - padding_before
                    padding = [(0, 0)] * i + [(padding_before, padding_after)] + [(0, 0)] * (len(arr.shape) - i - 1)
                    adjusted = np.pad(adjusted, padding, mode='constant', constant_values=0)
                elif arr.shape[i] > self.target_shape[i]:
                    # Cropping
                    start = (arr.shape[i] - self.target_shape[i]) // 2
                    end = start + self.target_shape[i]
                    adjusted = adjusted.take(indices=range(start, end), axis=i)
            return adjusted

        def __call__(self, sample):
            sample['f_shape'] = sample['f_img'].shape
            sample['m_shape'] = sample['m_img'].shape
            for key in sample:
                if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                    sample[key] = self.adjust_to_target_shape(sample[key])
            return sample

    class DatatypeConversion(object):
        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'm_img', 'f_seg', 'f_mask', 'm_seg', 'm_mask', 'f_input', 'm_input']:
                    sample[key] = sample[key].astype('float32')
            return sample

    class RangeNormalize(object):
        ''' Normalize the intensity values to [0, 1].
            Only applied to the images but not the inputs because inputs will be normalized by
            InstanceNorm within the network
        '''
        def __init__(self, min_percentile=0.01, max_percentile=0.99):
            self.max_percentile = max_percentile
            self.min_percentile = min_percentile

        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'm_img']:
                    min_intensity = np.percentile(sample[key], self.min_percentile)
                    max_intensity = np.percentile(sample[key], self.max_percentile)
                    sample[key] = np.clip(sample[key], min_intensity, max_intensity)
                    sample[key] = sample[key] - sample[key].min()
                    sample[key] = sample[key] / sample[key].max()
            return sample

    class Normalize(object):
        def __init__(self, f_scaling_factor, m_scaling_factor):
            self.f_scaling_factor = f_scaling_factor
            self.m_scaling_factor = m_scaling_factor

        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'f_input']:
                    sample[key] = sample[key] * self.f_scaling_factor
                elif key in ['m_img', 'm_input']:
                    sample[key] = sample[key] * self.m_scaling_factor

            return sample

    class ApplyMask(object):
        '''
            Apply mask to the images.
            This affects the loss computation.
        '''
        def __call__(self, sample):
            if 'f_mask' in sample:
                sample['f_img'] = sample['f_img'] * sample['f_mask']

            if 'm_mask' in sample:
                sample['m_img'] = sample['m_img'] * sample['m_mask']
            return sample

    class RandomInputMask(object):
        '''
            Apply mask to the input of the networks.
        '''
        def __call__(self, sample):
            if 'f_mask' in sample and randint(0, 1):
                sample['f_input'] = sample['f_input'] * sample['f_mask']

            if 'm_mask' in sample and randint(0, 1):
                sample['m_input'] = sample['m_input'] * sample['m_mask']
            return sample

    class RandomFlip3d(object):
        '''External affine grid needs to be adjusted if random flipping is applied'''
        def __call__(self, sample):
            if randint(0,1):
                for key in sample:
                    if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                        sample[key] = np.flip(sample[key], axis=0).copy()
                    if key in ['ext_affine_grid']:
                        sample[key] = torch.flip(sample[key], dims=[1])
                        sample[key][0, :, :, :] = sample[key].shape[1] - sample[key][0, :, :, :] - 1

            if randint(0,1):
                for key in sample:
                    if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                        sample[key] = np.flip(sample[key], axis=1).copy()
                    if key in ['ext_affine_grid']:
                        sample[key] = torch.flip(sample[key], dims=[2])
                        sample[key][1, :, :, :] = sample[key].shape[2] - sample[key][1, :, :, :] - 1

            if randint(0,1):
                for key in sample:
                    if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                        sample[key] = np.flip(sample[key], axis=2).copy()
                    if key in ['ext_affine_grid']:
                        sample[key] = torch.flip(sample[key], dims=[3])
                        sample[key][2, :, :, :] = sample[key].shape[3] - sample[key][2, :, :, :] - 1

            return sample

    class RandomFlip2d(object):
        def __call__(self, sample):
            if randint(0,1):
                for key in sample:
                    if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                        sample[key] = np.flip(sample[key], axis=0).copy()

            if randint(0,1):
                for key in sample:
                    if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                        sample[key] = np.flip(sample[key], axis=1).copy()

            return sample

    class RandomDirection(object):
        def __call__(self, sample):
            out = {}
            if randint(0, 1):
                for key in sample:
                    if key.startswith('f_'):
                        new_key = key.replace('f_', 'm_')
                        out[new_key] = sample[key]
                    elif key.startswith('m_'):
                        new_key = key.replace('m_', 'f_')
                        out[new_key] = sample[key]
                    else:
                        out[key] = sample[key]
                return out
            else:
                return sample

    class ToTensor(object):
        def __call__(self,sample):
            for key in sample:
                if key in ['f_img', 'm_img', 'f_seg', 'm_seg', 'f_mask', 'm_mask', 'f_input', 'm_input']:
                    sample[key] = torch.from_numpy(np.ascontiguousarray(sample[key].copy()))
                    sample[key] = sample[key][None,...]
            return sample

    class RandomAffine(object):
        '''
            Apply random affine transformation to the fixed or moving image.
            fm specifies whether to apply the transform to the fixed or moving
            rotation is specified in unit of pi
        '''
        def __init__(self, fm,
                    s_x_min=1, s_x_max=1, s_y_min=1, s_y_max=1, s_z_min=1, s_z_max=1,
                    r_xy_min=0, r_xy_max=0, r_yz_min=0, r_yz_max=0, r_xz_min=0, r_xz_max=0,
                    t_x_min=0, t_x_max=0, t_y_min=0, t_y_max=0, t_z_min=0, t_z_max=0
        ):

            self.fm = fm

            self.s_x_min = s_x_min
            self.s_y_min = s_y_min
            self.s_z_min = s_z_min
            self.s_x_max = s_x_max
            self.s_y_max = s_y_max
            self.s_z_max = s_z_max

            self.r_xy_min = r_xy_min
            self.r_yz_min = r_yz_min
            self.r_xz_min = r_xz_min
            self.r_xy_max = r_xy_max
            self.r_yz_max = r_yz_max
            self.r_xz_max = r_xz_max

            self.t_x_min = t_x_min
            self.t_y_min = t_y_min
            self.t_z_min = t_z_min
            self.t_x_max = t_x_max
            self.t_y_max = t_y_max
            self.t_z_max = t_z_max

        def __call__(self, sample):
            syn_trans = self.rd_affine_params()
            # create the synthetic image by sampling the original using the forward grid
            # the warped image should be created by sample the synthetic image
            # therefore, the inverse transformation is needed
            grid_forward, grid_inv = self.affine_grid_from_trans(syn_trans, sample[f'{self.fm}_img'])
            sample[f'{self.fm}_grid'] = grid_inv.squeeze(0) # needs to be adjusted when used with other preprocessing

            for key in sample:
                if key in [f'{self.fm}_seg', f'{self.fm}_mask']:
                    x = (torch.from_numpy(sample[key])[None, None, ...]).type(torch.float32)
                    sample[key] = grid_sampler(x, grid_forward, mode='nearest').squeeze().numpy()

                elif key in [f'{self.fm}_img', f'{self.fm}_input']:
                    x = (torch.from_numpy(sample[key])[None, None, ...]).type(torch.float32)
                    sample[key] = grid_sampler(x, grid_forward, mode='bilinear').squeeze().numpy()

            return sample

        def rd_affine_params(self):
            syn_trans = {} # synthetic transformation
            syn_trans['theta_xy'] = random.uniform(np.pi*self.r_xy_min, np.pi*self.r_xy_max)
            syn_trans['theta_yz'] = random.uniform(np.pi*self.r_yz_min, np.pi*self.r_yz_max)
            syn_trans['theta_xz'] = random.uniform(np.pi*self.r_xz_min, np.pi*self.r_xz_max)
            syn_trans['scaling'] = [
                                random.uniform(self.s_x_min, self.s_x_max),
                                random.uniform(self.s_y_min, self.s_y_max),
                                random.uniform(self.s_z_min, self.s_z_max)
            ]
            syn_trans['translation'] = [
                                random.uniform(self.t_x_min, self.t_x_max),
                                random.uniform(self.t_y_min, self.t_y_max),
                                random.uniform(self.t_z_min, self.t_z_max),
            ]
            return syn_trans

        def affine_grid_from_trans(self, syn_trans, reference):

            def numpy2torch(matrix):
                return (torch.from_numpy(matrix)[None,...]).type(torch.float32)

            # Rotation matrix in the xy-plane
            theta_xy = syn_trans['theta_xy']
            R_xy = np.array([
                [np.cos(theta_xy), -np.sin(theta_xy), 0, 0],
                [np.sin(theta_xy),  np.cos(theta_xy), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # Rotation matrix in the yz-plane
            theta_yz = syn_trans['theta_yz']
            R_yz = np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta_yz), -np.sin(theta_yz), 0],
                [0, np.sin(theta_yz),  np.cos(theta_yz), 0],
                [0, 0, 0, 1]
            ])

            # Rotation matrix in the xz-plane
            theta_xz = syn_trans['theta_xz']
            R_xz = np.array([
                [np.cos(theta_xz), 0, np.sin(theta_xz), 0],
                [0, 1, 0, 0],
                [-np.sin(theta_xz), 0, np.cos(theta_xz), 0],
                [0, 0, 0, 1]
            ])

            # Combined rotation matrix
            R = R_xy @ R_yz @ R_xz

            # scaling
            scaling_x, scaling_y, scaling_z = syn_trans['scaling']
            S = np.array([
                [scaling_x, 0, 0, 0],
                [0, scaling_y, 0, 0],
                [0, 0, scaling_z, 0],
                [0, 0, 0, 1]
            ])

            # add translation
            translation_x, translation_y, translation_z = syn_trans['translation']
            T = np.array([
                [1, 0, 0, translation_x],
                [0, 1, 0, translation_y],
                [0, 0, 1, translation_z],
                [0, 0, 0, 1]
            ])

            # center_alignment for applying rotation at the center of image
            center_align_matrix = np.array([
                [1, 0, 0, -int(reference.shape[0] / 2)],
                [0, 1, 0, -int(reference.shape[1] / 2)],
                [0, 0, 1, -int(reference.shape[1] / 2)],
                [0, 0, 0, 1]
            ])
            # inverse the center alignment
            inv_center_align_matrix = np.linalg.inv(center_align_matrix)

            reference = torch.from_numpy(reference)[None, None, ...]
            reference = reference.type(torch.float32)
            X = identity_grid_like(reference, normalize=False)
            shape = X.shape
            X = X.flatten(start_dim=2) # Bx3x(HxWxD)
            X = torch.cat((X, torch.ones_like(X[:, :1, :])), dim=1) # add homogeneous coordinates

            # Y = torch.matmul(numpy2torch(center_align_matrix), X)
            # Y = torch.matmul(numpy2torch(S), Y)
            # Y = torch.matmul(numpy2torch(R), Y)
            # Y = torch.matmul(numpy2torch(inv_center_align_matrix), Y)
            # Y = torch.matmul(numpy2torch(T), Y)

            # combine the matrices first is more efficient
            combined_matrix = T @ inv_center_align_matrix @ R @ S @ center_align_matrix
            Y = torch.matmul(numpy2torch(combined_matrix), X)
            Y = Y[:, :3, :]
            Y = Y.view(shape)

            inv_combined_matrix = np.linalg.inv(combined_matrix)
            Z = torch.matmul(numpy2torch(inv_combined_matrix), X)
            Z = Z[:, :3, :]
            Z = Z.view(shape)

            return Y, Z

