import os
import nibabel as nib
import pdb
import numpy as np
import torch
import random
import pathlib

from vfa.datasets import BaseDataset
from vfa.datasets.pairwise_dataset import PairwiseDataset

class L2R2022NLSTDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)

    def __len__(self):
        return len(self.configs['pairs'])

    def __getitem__(self, k):
        data_path = self.configs['pairs'][k]
        sample = {}

        sample['f_img_path'] = os.path.abspath(data_path['f_img'])
        sample['f_img'] = self.load_img_obj(sample['f_img_path']) 

        sample['m_img_path'] = os.path.abspath(data_path['m_img'])
        sample['m_img'] = self.load_img_obj(sample['m_img_path']) 

         # if relative path is provided as prefix, save the results to ./experiment
         if os.path.isabs(data_path['prefix']):
             sample['prefix'] = data_path['prefix']
         else:
             sample['prefix'] = os.path.abspath(os.path.join(
                                    self.params['cwd'],
                                    'experiments',
                                    'l2r2022nlst',
                                    data_path['prefix']
             ))


        if 'f_mask' in data_path:
            sample['f_mask_path'] = os.path.abspath(data_path['f_mask'])
            sample['f_mask'] = self.load_img_obj(sample['f_mask_path'])

        if 'm_mask' in data_path:
            sample['m_mask_path'] = os.path.abspath(data_path['m_mask'])
            sample['m_mask'] = self.load_img_obj(sample['m_mask_path'])

        if 'f_seg' in data_path:
            sample['f_seg_path'] = os.path.abspath(data_path['f_seg'])
            sample['f_seg'] = self.load_img_obj(sample['f_seg_path'])

        if 'm_seg' in data_path:
            sample['m_seg_path'] = os.path.abspath(data_path['m_seg'])
            sample['m_seg'] = self.load_img_obj(sample['m_seg_path'])

        if 'f_keypoints' in data_path:
            sample['f_keypoints_path'] = os.path.abspath(data_path['f_keypoints'])
            sample['f_keypoints'] = np.genfromtxt(sample['f_keypoints_path'], delimiter=',')

        if 'm_keypoints' in data_path:
            sample['m_keypoints_path'] = os.path.abspath(data_path['m_keypoints'])
            sample['m_keypoints'] = np.genfromtxt(sample['m_keypoints_path'], delimiter=',')

        sample['sid'] = data_path['sid']

        sample = self.transforms(sample)

        return sample

    class DatatypeConversion(object):
        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'm_img', 'f_seg', 'f_mask', 'm_seg', 'm_mask']:
                    sample[key] = sample[key].astype('float32')
                if key in ['f_keypoints', 'm_keypoints']:
                    sample[key] = sample[key].astype('float32')
            return sample

    class RandomFlip3d(object):
        def __call__(self, sample):
            if random.randint(0,1):
                for key in sample:
                    if key in ['f_img', 'f_mask', 'm_img', 'm_mask']:
                        sample[key] = np.flip(sample[key], axis=0).copy()
                    if key in ['f_keypoints', 'm_keypoints']:
                        sample[key][:,0] = 223 - sample[key][:,0]
            if random.randint(0,1):
                for key in sample:
                    if key in ['f_img', 'f_mask', 'm_img', 'm_mask']:
                        sample[key] = np.flip(sample[key], axis=1).copy()
                    if key in ['f_keypoints', 'm_keypoints']:
                        sample[key][:,1] = 191 - sample[key][:,1]
            if random.randint(0,1):
                for key in sample:
                    if key in ['f_img', 'f_mask', 'm_img', 'm_mask']:
                        sample[key] = np.flip(sample[key], axis=2).copy()
                    if key in ['f_keypoints', 'm_keypoints']:
                        sample[key][:,2] = 223 - sample[key][:,2]
            return sample

    class ToTensor(object):
        def __call__(self, sample):
            for key in sample:
                if key in ['f_img', 'f_mask', 'm_img', 'm_mask']:
                    sample[key] = torch.from_numpy(np.ascontiguousarray(sample[key].copy())).type(torch.float32)
                    sample[key] = sample[key][None,...]
                if key in ['f_keypoints', 'm_keypoints']:
                    sample[key] = torch.from_numpy(np.ascontiguousarray(sample[key].copy())).type(torch.float32)
                    # this is use in the original code because of the grid_sample function implemented in pytorch. 
                    # to prevent misinterpretation of the dimensions, we remove it here and add it back when we need to use the grid_sample function
                    # sample[key] = sample[key].flip(-1)
            return sample 

    def export(self, results, sample):
        if self.params['save_results'] == 0:
            return None
        super().export(results, sample)

        # -------------------- for learn2reg 2022 task1 submission----------------------
        from utils.utils import identity_grid_like

        disp = results['grid'] - identity_grid_like(results['grid'], normalize=False)
        disp = disp.detach().permute(0, 2, 3, 4, 1).cpu().squeeze().numpy()

        submission_path = pathlib.Path(self.params['cwd']) / 'experiments' / 'l2r2022nlst' / 'nlst_zerofield' / 'nlst_zerofield' / 'test_zf'
        submission_path.mkdir(exist_ok=True, parents=True)
        filename = f"disp_{sample['sid'][0]}_{sample['sid'][0]}.nii.gz"
        nib.save(nib.Nifti1Image(disp, np.eye(4)), str(submission_path / filename))
