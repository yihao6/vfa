import os
import pdb
import random

from vfa.datasets import BaseDataset
from vfa.utils.utils import calc_affine_grid, get_ants_affine_matrix

class InterDataset(BaseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)
        # if moving samples are not specified, register within the fixed samples
        if 'm_subjects' not in self.configs:
            self.configs['m_subjects'] = self.configs['f_subjects']

    def __len__(self):
        return 200

    def __getitem__(self, _):
        sample = {}

        # register randomly selected samples from fixed and moving list
        f_sample = random.choice(self.configs['f_subjects'])
        sample['f_img_path'] = os.path.abspath(f_sample['img'])
        sample['f_img'] = self.load_img_obj(sample['f_img_path'])

        m_sample = random.choice(self.configs['m_subjects'])
        sample['m_img_path'] = os.path.abspath(m_sample['img'])
        sample['m_img'] = self.load_img_obj(sample['m_img_path'])

        if 'inputs' in f_sample:
            sample['f_input_path'] = os.path.abspath(random.choice(f_sample['inputs']))
            sample['f_input'] = self.load_img_obj(sample['f_input_path'])

        if 'inputs' in m_sample:
            sample['m_input_path'] = os.path.abspath(random.choice(m_sample['inputs']))
            sample['m_input'] = self.load_img_obj(sample['m_input_path'])

        if 'mask' in f_sample:
            sample['f_mask_path'] = os.path.abspath(f_sample['mask'])
            sample['f_mask'] = self.load_img_obj(sample['f_mask_path'])

        if 'mask' in m_sample:
            sample['m_mask_path'] = os.path.abspath(m_sample['mask'])
            sample['m_mask'] = self.load_img_obj(sample['m_mask_path'])

        if 'seg' in f_sample:
            sample['f_seg_path'] = os.path.abspath(f_sample['seg'])
            sample['f_seg'] = self.load_img_obj(sample['f_seg_path'])

        if 'seg' in m_sample:
            sample['m_seg_path'] = os.path.abspath(m_sample['seg'])
            sample['m_seg'] = self.load_img_obj(sample['m_seg_path'])

        if 'prefix' in f_sample:
            # if relative path is provided as prefix, save the results to ./experiment
            if os.path.isabs(f_sample['prefix']):
                sample['prefix'] = f_sample['prefix']
            else:
                sample['prefix'] = os.path.abspath(os.path.join(
                                        self.params['cwd'],
                                        'experiments',
                                        f_sample['prefix']
                ))
        elif 'prefix' in m_sample:
            # if relative path is provided as prefix, save the results to ./experiment
            if os.path.isabs(m_sample['prefix']):
                sample['prefix'] = m_sample['prefix']
            else:
                sample['prefix'] = os.path.abspath(os.path.join(
                                        self.params['cwd'],
                                        'experiments',
                                        m_sample['prefix']
                ))

        if 'ext_affine_matrix' in f_sample:
            ants_affine_matrix = get_ants_affine_matrix(f_sample['ext_affine_matrix'])
            sample['ext_affine_grid'] = calc_affine_grid(
                    ants_affine_matrix,
                    sample['f_img'].affine,
                    sample['m_img'].affine, # m_input has been affinely registered
                    self.configs['shape'][1:],
            )
        elif 'ext_affine_matrix' in m_sample:
            ants_affine_matrix = get_ants_affine_matrix(m_sample['ext_affine_matrix'])
            sample['ext_affine_grid'] = calc_affine_grid(
                    ants_affine_matrix,
                    sample['f_img'].affine,
                    sample['m_img'].affine, # m_input has been affinely registered
                    self.configs['shape'][1:],
            )

        sample = self.transforms(sample)

        return sample
