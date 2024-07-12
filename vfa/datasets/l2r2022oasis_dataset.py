import pdb
import numpy as np
from scipy.ndimage import zoom
import pathlib

from vfa.utils.utils import identity_grid_like, grid_sampler
from vfa.datasets.pairwise_dataset import PairwiseDataset

class L2R2022OASISDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)

    def __getitem__(self, k):
        sample = super().__getitem__(k)
        data_path = self.configs['pairs'][k]
        sample['f_sid'] = data_path['f_sid']
        sample['m_sid'] = data_path['m_sid']

        # if relative path is provided as prefix, save the results to ./experiment
        if os.path.isabs(data_path['prefix']):
            sample['prefix'] = data_path['prefix']
        else:
            sample['prefix'] = os.path.abspath(os.path.join(
                                self.params['output_dir'],
                                'experiments',
                                'l2r2022oasis',
                                data_path['prefix']
            ))

        return sample

    def export(self, results, sample): 
        if self.params['save_results'] == 0:
            return None
        super().export(results, sample)

        '''for learn2reg 2021 task3 submission'''
        disp = results['grid'] - identity_grid_like(results['grid'], normalize=False)
        # L2R2021 Task3 use half percision format
        disp = disp.detach().cpu().numpy()[0]
        disp_half = np.array([zoom(disp[x], 0.5, order=2) for x in range(3)])
        disp_half = disp_half.astype('float16')
        submission_path = pathlib.Path(self.params['output_dir']) / 'experiments' / 'l2r2022oasis'/ 'submission' / 'task_03'
        submission_path.mkdir(exist_ok=True, parents=True)
        filename = f"disp_{sample['f_sid'][0]}_{sample['m_sid'][0]}.npz"
        np.savez(str(submission_path / filename), disp_half)

        # dummy txt file for online evaluation system
        dummy_path = submission_path.parents[0] / 'dummy.txt'
        if not dummy_path.exists():
            with open(str(dummy_path), 'w'):
                pass
