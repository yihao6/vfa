import os
import pdb

from vfa.datasets.pairwise_dataset import PairwiseDataset

class CLDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__()

        self.configs['pairs'] = [{'id': 1}]
        for key in ['prefix', 'f_img', 'm_img', 'f_input', 'm_input',
            'f_mask', 'm_mask', 'f_seg', 'm_seg']:
            if key not in params:
                raise KeyError(f'Argument {key} not provided in the command line.')
            else:
                self.configs['pairs'][0][key] = params[key]
