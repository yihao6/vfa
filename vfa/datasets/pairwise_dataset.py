import os
import pdb
import random

from vfa.datasets import BaseDataset

class PairwiseDataset(BaseDataset):
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
        if 'prefix' in data_path:
            if os.path.isabs(data_path['prefix']):
                sample['prefix'] = data_path['prefix']
            else:
                sample['prefix'] = os.path.abspath(os.path.join(
                                        self.params['cwd'],
                                        'experiments',
                                        self.params['identifier'],
                                        data_path['prefix']
                ))

        # 
        if 'f_inputs' in data_path:
            sample['f_input_path'] = os.path.abspath(random.choice(data_path['f_inputs']))
            sample['f_input'] = self.load_img_obj(sample['f_input_path'])

        if 'm_inputs' in data_path:
            sample['m_input_path'] = os.path.abspath(random.choice(data_path['m_inputs']))
            sample['m_input'] = self.load_img_obj(sample['m_input_path'])

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

        if 'ext_affine_matrix' in data_path:
            ants_affine_matrix = get_ants_affine_matrix(data_path['ext_affine_matrix'])
            sample['ext_affine_grid'] = calc_affine_grid(
                    ants_affine_matrix,
                    sample['f_img'].affine,
                    sample['m_img'].affine, # m_input has been affinely registered
                    self.configs['shape'][1:],
            )

        sample = self.transforms(sample)

        return sample
