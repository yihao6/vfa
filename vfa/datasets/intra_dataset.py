import os
import random
import pdb

from vfa.datasets import BaseDataset

class IntraDataset(BaseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)

    def __len__(self):
        return 200

    def __getitem__(self, k):
        sample = {}
        subject = random.choice(self.configs['subjects'])
        f_sample, m_sample = random.sample(subject, k=2)

        sample['f_path'] = os.path.abspath(f_sample['img'])
        sample['f_img'] = self.load_img_obj(sample['f_path'])

        sample['m_path'] = os.path.abspath(m_sample['img'])
        sample['m_img'] = self.load_img_obj(sample['m_path'])

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

        sample = self.transforms(sample)

        return sample
