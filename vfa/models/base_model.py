import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
import pdb
import numpy as np
import logging

from vfa.utils.utils import onehot, grid_sampler, identity_grid_like, keypoints_sampler, normalize_grid
from vfa.losses import create_loss_instance
from vfa.utils.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = params
        self.device = device

        self.init_losses()

    def init_losses(self):
        # instanize loss object
        self.loss_instances = {}
        exceptions = ['TRE', 'Beta', 'Grid'] # not implemented in a separate file

        for phase in self.params['loss']:
            loss_names = list(self.params['loss'][phase].keys())
            loss_names = [loss for loss in loss_names if loss not in exceptions]

            self.loss_instances[phase] = nn.ModuleDict({
                loss_name: create_loss_instance(
                                loss_name.replace('Affine_', ''),
                                **self.params['loss'][phase].get(loss_name, {})
                )
                for loss_name in loss_names
            })

    def generate_results(self, grid, sample):
        if 'm_mask' not in sample:
            sample['m_mask'] = torch.ones_like(sample['m_img'])
        if 'f_mask' not in sample:
            sample['f_mask'] = torch.ones_like(sample['f_img'])

        # record the initial output
        results = {'grid': grid}
        # compose the external affine grid if provided
        if 'ext_affine_grid' in sample:
            sample['ext_affine_grid'] = sample['ext_affine_grid'].to(self.device)
            grid = grid_sampler(sample['ext_affine_grid'], grid)

        w_img = grid_sampler(sample['m_img'].to(self.device), grid)
        w_mask = grid_sampler(
                        sample['m_mask'].to(self.device),
                        grid,
                        mode='nearest',
                        padding_mode='zeros',
        )
        mask = sample['f_mask'].to(self.device) * w_mask
        if torch.any(torch.sum(mask, dim=tuple(range(1, mask.dim()))) == 0):
            raise ValueError('No overlap between warped and fixed image masks.')

        if 'f_keypoints' in sample:
            w_keypoints = keypoints_sampler(grid, sample['f_keypoints'].to(self.device))
            w_keypoints = w_keypoints.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        else:
            w_keypoints = None

        results.update({
                'w_img': w_img,
                'w_mask':w_mask,
                'w_keypoints':w_keypoints,
                'mask': mask, # combined mask
        })
        return results

    def generate_affine_results(self, grid, sample):
        raise NotImplementedError('To be released with VFA2.')

    def calc_dice_loss(self, results, sample, phase, labels, **args):
        if 'f_seg' not in sample or 'm_seg' not in sample:
            logger.warning('Label maps not available for calculating Dice')
            return torch.tensor(0.0).to(self.device)

        # compose the external affine grid if provided
        if 'ext_affine_grid' in sample:
            sample['ext_affine_grid'] = sample['ext_affine_grid'].to(self.device)
            grid = grid_sampler(sample['ext_affine_grid'], results['grid'])
        else:
            grid = results['grid']

        if phase == 'train' and 'Dice' in self.params['loss']['train']:
            f_onehot = onehot(sample['f_seg'].to(self.device), labels)
            m_onehot = onehot(sample['m_seg'].to(self.device), labels)
            w_onehot = grid_sampler(m_onehot, grid)
            with autocast(enabled=False):
                loss = self.loss_instances[phase]['Dice'](
                                    w_onehot.type(torch.float32),
                                    f_onehot.type(torch.float32)
                )
            return torch.mean(loss) # average across labels
        else:
            '''report the real Dice during evaluation, not Dice loss (1 - Dice)'''
            # when use nearest interpolation, convert to onehot can be done later
            w_seg = grid_sampler(
                            sample['m_seg'].to(self.device),
                            grid,
                            mode='nearest',
            )

            # GPU implementation, faster but takes more GPU memory
            w_onehot = onehot(w_seg, labels)
            f_onehot = onehot(sample['f_seg'].to(self.device), labels)
            loss = self.loss_instances[phase]['Dice'](w_onehot, f_onehot)
            return torch.mean(1 - loss)

            # CPU implementation
            def calc_dice(pred, reference, labels):
                dice = 0
                for label in labels:
                    label_dice = np.sum(reference[pred == label] == label) * 2.0 / (np.sum(pred == label) + np.sum(reference == label))
                    dice += label_dice
                return dice / len(labels)

            dice_score = calc_dice(
                                w_seg.cpu().numpy().squeeze(),
                                sample['f_seg'].numpy().squeeze(),
                                labels,
            )
            return torch.tensor(dice_score)

    def calc_mse_loss(self, results, sample, phase, **args):
        pred = results['w_img'] * results['mask']
        target = sample['f_img'].to(self.device) * results['mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['MSE'](pred, target)

        return loss

    def calc_rngf_loss(self, results, sample, phase, **args):
        pred = results['w_img']
        target = sample['f_img'].to(self.device)

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['RNGF'](pred, target, results['mask'])
        return loss

    def calc_corrratio_loss(self, results, sample, phase, **args):
        pred = results['w_img'] * results['mask']
        target = sample['f_img'].to(self.device) * results['mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['CorrRatio'](pred, target)
        return loss

    def calc_mi_loss(self, results, sample, phase, **args):
        pred = results['w_img'] * results['mask']
        target = sample['f_img'].to(self.device) * results['mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['MI'](pred, target)
        return loss

    def calc_msmi_loss(self, results, sample, phase, **args):
        pred = results['w_img'] * results['mask']
        target = sample['f_img'].to(self.device) * results['mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['MSMI'](pred, target)
        return loss

    def calc_affine_mi_loss(self, results, sample, phase, **args):
        pred = results['affine_w_img'] * results['affine_mask']
        target = sample['f_img'].to(self.device) * results['affine_mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['Affine_MI'](pred, target)
        return loss

    def calc_affine_msmi_loss(self, results, sample, phase, **args):
        pred = results['affine_w_img'] * results['affine_mask']
        target = sample['f_img'].to(self.device) * results['affine_mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['Affine_MSMI'](pred, target)
        return loss

    def calc_ncc_loss(self, results, sample, phase, **args):
        pred = results['w_img'] * results['mask']
        target = sample['f_img'].to(self.device) * results['mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['NCC'](pred, target)
        return loss

    def calc_affine_ncc_loss(self, results, sample, phase, **args):
        pred = results['affine_w_img'] * results['affine_mask']
        target = sample['f_img'].to(self.device) * results['affine_mask']

        with autocast(enabled=False):
            pred = pred.type(torch.float32)
            target = target.type(torch.float32)
            loss = self.loss_instances[phase]['Affine_NCC'](pred, target)
        return loss

    def calc_bending_loss(self, results, sample, phase, **args):
        # default value for voxel resolution
        B = results['grid'].shape[0]
        voxel_dims = torch.tensor([1.0, 1.0, 1.0]).expand(B, 3).to(self.device)

        loss = self.loss_instances[phase]['Bending'](results['grid'], voxel_dims)
        return loss

    def calc_grad_loss(self, results, phase, **args):
        identity = identity_grid_like(results['grid'], normalize=False)
        disp = results['grid'] - identity
        loss = self.loss_instances[phase]['Grad'](disp)

        return loss

    def calc_affine_grad_loss(self, results, phase, **args):
        if not self.params['model']['affine']:
            raise ValueError('affine is set to False in Model, affine grid loss unavailable.')
        nonlinear_grid = grid_sampler(
                    results['inv_affine_grid'],
                    results['grid'],
        )
        disp = nonlinear_grid - identity_grid_like(nonlinear_grid, normalize=False)
        loss = self.loss_instances[phase]['Affine_Grad'](disp)

        return loss

    def calc_tre_loss(self, results, sample, phase, **args):
        '''Target error loss computations are different for each dataset'''
        if 'm_keypoints' not in sample or results['w_keypoints'] is None:
            # keypoints not available
            return torch.tensor(0.0).to(self.device)

        target = sample['m_keypoints'].to(self.device)
        pred = results['w_keypoints']
        if self.params['loss'][phase]['TRE']['method'] == 'L2R2022NLST':
            return (pred - target).squeeze().pow(2).sum(-1).sqrt().mean()*1.5 # 1.5 is resolution
        elif self.params['loss'][phase]['TRE']['method'] == 'BraTSReg2022':
            return (pred - target).squeeze().abs().sum(-1).mean() # MAE
        else:
            return (pred - target).squeeze().pow(2).sum(-1).sqrt().mean() # RMSE

    def calc_grid_loss(self, results, sample, phase, **args):
        '''
            Direct supervision on the sampling grid.
            Usually available in simulated transformation.
        '''
        if 'm_grid' not in sample:
            return torch.tensor(0.0).to(self.device)
        target = normalize_grid(sample['m_grid']).to(self.device)
        pred = normalize_grid(results['grid'])
        return (pred - target).squeeze().pow(2).mean()

    def print_num_parameters(self):
        # number of weights and bias
        num_weights = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and name.endswith(".weight"))
        num_biases = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and name.endswith(".bias"))
        logger.info(f"Weights: {num_weights}, Bias: {num_biases}, Total: {num_weights+num_biases}")

    def print_flops(self):
        from thop import profile
        # create dummy input
        if len(self.params['model']['in_shape']) == 3:
            logger.info("---- Input image shape: [1, 192, 224, 192].")
            shape = [1, 192, 224, 192]
        elif len(self.params['model']['in_shape']) == 2:
            shape = [1, 512, 512]
        sample = {
            'f_img': torch.randn(1, *shape).to(self.device),
            'm_img': torch.randn(1, *shape).to(self.device),
            'f_mask': torch.ones(1, *shape).to(self.device),
            'm_mask': torch.ones(1, *shape).to(self.device),
        }

        flops, params = profile(self, inputs=(sample,))
        def human_readable_flops(num):
            if num < 1e6:  # less than a million
                return f"{num:.2f} FLOPs"
            elif num < 1e9:  # less than a billion
                return f"{num / 1e6:.2f} million FLOPs"
            elif num < 1e12:  # less than a trillion
                return f"{num / 1e9:.2f} billion FLOPs"
            else:  # trillion or more
                return f"{num / 1e12:.2f} trillion FLOPs"

        def human_readable_number(num):
            for unit in ['', 'K', 'M', 'G', 'T']:
                if abs(num) < 1000.0:
                    return f"{num:3.2f}{unit}"
                num /= 1000.0
            return f"{num:.2f}P"  # 'P' for Peta

        # flops_readable = human_readable_flops(flops)
        flops_readable = human_readable_number(flops)
        params_readable = human_readable_number(params)  # You can use the previous function for parameters as it's typically in terms of 'M' or 'G'.

        logger.info(f'FLOPs: {flops_readable}, Number of Parameters: {params_readable}')

    def test_gradient_nan(self):
        # for debug purpuse
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                if torch.any(torch.isnan(parameter.grad)):
                    logger.info(f"NaN gradient in {name}")

