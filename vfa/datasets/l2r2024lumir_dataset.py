import pdb
import numpy as np
from scipy.ndimage import zoom
import re
import os
import nibabel as nib
import pathlib

from vfa.utils.utils import identity_grid_like, grid_sampler
from vfa.datasets.pairwise_dataset import PairwiseDataset

class L2R2024LUMIRDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)

    def __getitem__(self, k):
        sample = super().__getitem__(k)
        data_path = self.configs['pairs'][k]
        sid_regex = re.compile(r'LUMIRMRI_(\d+)_0000.nii.gz')
        f_sid = sid_regex.search(os.path.basename(data_path['f_img'])).group(1)
        m_sid = sid_regex.search(os.path.basename(data_path['m_img'])).group(1)

        sample['filename'] = f'disp_{f_sid}_{m_sid}.nii.gz'
        sample['prefix'] = os.path.abspath(os.path.join(
                                self.params['cwd'],
                                'experiments',
                                'l2r2024lumir',
                                sample['filename']
        ))

        return sample

    def export(self, results, sample):
        if self.params['save_results'] == 0:
            return None

        super().export(results, sample)

        '''for L2R 2024 LUMIR submission'''
        disp = results['grid'] - identity_grid_like(results['grid'], normalize=False)
        disp = disp.detach().cpu().numpy()[0].transpose(1, 2, 3, 0)

        submission_path = pathlib.Path(self.params['cwd']) / 'experiments' / 'l2r2024lumir' / 'submission'
        submission_path.mkdir(exist_ok=True, parents=True)

        ref_obj = nib.load(sample['f_img_path'][0])
        nib.save(nib.Nifti1Image(disp, ref_obj.affine), str(submission_path / sample['filename'][0]))

