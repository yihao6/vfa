import torch.nn as nn
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import socket
import os
import numpy as np
import pathlib
import argparse
import json
import pdb
import logging
from datetime import datetime
from tqdm import tqdm
from contextlib import nullcontext

from vfa.models import create_network_class
from vfa.datasets import load_dataset
from vfa.utils.utils import update_params_json, update_params_args, setup_logging, load_data_configs
setup_logging()
logger = logging.getLogger(__name__)

@torch.inference_mode(False)
def train(net, trainDL, params, optimizer, scheduler, scaler, pbar, data_configs):
    net.train()
    epoch_loss = {x:[] for x in params['loss']['train']}

    for sample in trainDL:
        sample_loss = {x:0 for x in params['loss']['train']}

        with autocast(dtype=torch.float16) if params['fp16'] else nullcontext():
            results = net(sample)
            total_loss = 0
            for loss in params['loss']['train']:
                loss_function_name = f"calc_{loss.lower()}_loss"
                loss_function = getattr(net, loss_function_name, None)
                if loss_function is not None:
                    sample_loss[loss] = loss_function(
                                                results=results,
                                                sample=sample,
                                                phase='train',
                                                labels=data_configs['labels']
                    )
                    sample_loss[loss] *= params['loss']['train'][loss]['weight']
                else:
                    raise NotImplementedError(f'{loss_function} not implemented')

                epoch_loss[loss].append(sample_loss[loss].item())
                if loss in params['loss']['train']:
                    total_loss += sample_loss[loss]

        optimizer.zero_grad()
        if params['fp16']:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        pbar.update(1)

    return epoch_loss

@torch.inference_mode()
def evaluate(net, evalDL, params, pbar, data_configs):
    net.eval()
    epoch_loss = {x:[] for x in params['loss']['eval']}

    for sample in evalDL:
        sample_loss = {x:0 for x in params['loss']['eval']}
        results = net(sample)
        for loss in params['loss']['eval']:
            loss_function_name = f"calc_{loss.lower()}_loss"
            loss_function = getattr(net, loss_function_name, None)
            if loss_function is not None:
                sample_loss[loss] = loss_function(
                                            results=results,
                                            sample=sample,
                                            phase='eval',
                                            labels=data_configs['labels']
                )
                sample_loss[loss] *= params['loss']['eval'][loss]['weight']
            else:
                raise NotImplementedError(f'{loss_function_name} not implemented')
            epoch_loss[loss].append(sample_loss[loss].cpu().item())

        evalDL.dataset.export(results, sample)
        pbar.update(1)
    return epoch_loss

def add_arguments(subparser):
    subparser.add_argument("--gpu", help="GPU ID", default=0)
    subparser.add_argument("--checkpoint", type=os.path.abspath, help="Path to a pretrained models")
    subparser.add_argument("--params", type=os.path.abspath, help="Specify params.json to load. Default: params.json in the checkpoints folder")
    subparser.add_argument("--eval_data_configs", type=os.path.abspath, help="Specify data_configs.json to load. Default: data_configs.json in the checkpoints folder")
    subparser.add_argument("--cudnn", action='store_true', default=False, help="Enable CUDNN")

def main():
    logger.info('Program started')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func='train')
    train_parser.add_argument("--output_dir", type=os.path.abspath, help="Output directory.", default='./vfa')
    train_parser.add_argument("--identifier", help="A string that identify the current run.", required=True)
    train_parser.add_argument("--train_data_configs", type=os.path.abspath, help="Specify data_configs.json to load for training. Default: data_configs.json in the checkpoints folder")
    train_parser.add_argument("--save_results", type=int, default=0)
    add_arguments(train_parser)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.set_defaults(func='evaluate')
    evaluate_parser.add_argument("--save_results", type=int, default=1)
    evaluate_parser.add_argument("--f_img", type=os.path.abspath, help="Path to the fixed image.")
    evaluate_parser.add_argument("--m_img", type=os.path.abspath, help="Path to the moving image.")
    evaluate_parser.add_argument("--f_input", type=os.path.abspath, help="Path to the fixed input image.")
    evaluate_parser.add_argument("--m_input", type=os.path.abspath, help="Path to the moving input image.")
    evaluate_parser.add_argument("--f_mask", type=os.path.abspath, help="Path to the fixed mask.")
    evaluate_parser.add_argument("--m_mask", type=os.path.abspath, help="Path to the moving mask.")
    evaluate_parser.add_argument("--f_seg", type=os.path.abspath, help="Path to the fixed label map.")
    evaluate_parser.add_argument("--m_seg", type=os.path.abspath, help="Path to the moving label map.")
    evaluate_parser.add_argument("--prefix", type=os.path.abspath, help="Prefix of the saved results.")
    add_arguments(evaluate_parser)

    args = parser.parse_args()

    '''load hyper parameters'''
    params = {}
    if args.checkpoint:
        # if use pretrained model is specified, load the previous hyper paremeters
        params_path = os.path.join(os.path.dirname(args.checkpoint), 'params.json')
        if os.path.exists(params_path):
            params = update_params_json(params_path, params)
        else:
            logger.warning(f'Cannot find params.json for the checkpoint at default path {params_path}')
    if args.params:
        # load the hyper parameters from json file
        params = update_params_json(args.params, params)
    params = update_params_args(args, params)

    eval_data_configs = load_data_configs(params['eval_data_configs'])
    params['model']['in_channels'] = eval_data_configs['shape'][0]
    params['model']['in_shape'] = eval_data_configs['shape'][1:]

    logger.info('List parameters')
    for item in params:
        logger.info(f'---- {item}:{params[item]}')

    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if params['cudnn']:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    logger.info('Load datasets')
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': False}
    eval_dataset_class = load_dataset(eval_data_configs['loader'])
    evalDS = eval_dataset_class(eval_data_configs, params)
    evalDL = torch.utils.data.DataLoader(evalDS, batch_size=1, shuffle=False, **kwargs)

    if params['func'] == 'train':
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}
        train_data_configs = load_data_configs(params['train_data_configs'])

        train_dataset_class = load_dataset(train_data_configs['loader'])
        trainDS = train_dataset_class(train_data_configs, params)
        trainDL = torch.utils.data.DataLoader(
                                trainDS,
                                batch_size=params['batch_size'],
                                shuffle=True,
                                **kwargs,
        )

    logger.info('Initialize network')
    net = create_network_class(params['model']['name'])(params, device)
    optimizer = optim.Adam(net.parameters(), lr=params['LR'], amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['LR_schedule'], gamma=0.5)
    scaler = GradScaler() if params['fp16'] else None

    '''load pretrained model'''
    if params['checkpoint']:
        if os.path.exists(params['checkpoint']): 
            checkpoint = torch.load(params['checkpoint'], map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint file {params['checkpoint']} does not exist.")
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        except KeyError:
            '''load pretrained model in other format'''
            start_epoch = net.load_pretrained_model(checkpoint, optimizer)
    else:
        start_epoch = 1

    if params['func'] == 'train':
        logger.info('Training started')
        # setting up tensorboard
        from torch.utils.tensorboard import SummaryWriter
        tb_folder = os.path.join(params['output_dir'], 'runs')
        tb_filename = params['identifier']+'-'+str(datetime.now())
        writer = SummaryWriter(os.path.join(tb_folder, tb_filename))
        logger.info(f'Tensorboard directory: {tb_folder}')

        # checkpoints
        checkpoint_folder = f"{params['output_dir']}/checkpoints/{params['identifier']}"
        pathlib.Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(checkpoint_folder, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4, separators=(',',':'))
        logger.info(f'Checkpoint directory: {checkpoint_folder}')

        # main progress bar
        main_pbar = tqdm(total=params['num_epochs'], position=0)
        main_pbar.set_description(f"[{socket.gethostname().split('.')[0]}-{params['gpu']}][{checkpoint_folder}]")
        eval_pbar = tqdm(total=len(evalDL), position=1)
        train_pbar = tqdm(total=len(trainDS) // params['batch_size'], position=2)

        for epoch in range(start_epoch, params['num_epochs']):
            '''run validation'''
            eval_pbar.reset()
            eval_pbar.set_description(f"[Evaluation][Epoch: {epoch}|{params['num_epochs']}]")
            eval_pbar.refresh()
            epoch_loss = evaluate(net, evalDL, params, eval_pbar, eval_data_configs)

            for key in epoch_loss:
                average_loss = np.mean(np.array(epoch_loss[key]))
                writer.add_scalar(f"validate/{key}-Loss", average_loss, epoch)

            '''save model'''
            save_model_name = str(epoch) + '-net.pth' if epoch % 50 == 0 else 'net.pth'
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(checkpoint_folder, save_model_name))

            '''training'''
            train_pbar.reset()
            train_pbar.set_description(f"[Training][Epoch: {epoch}|{params['num_epochs']}]")
            train_pbar.refresh()
            epoch_loss = train(net, trainDL, params, optimizer, scheduler, scaler, train_pbar, train_data_configs)

            for key in epoch_loss:
                average_loss = np.mean(np.array(epoch_loss[key]))
                writer.add_scalar(f"train/{key}-Loss", average_loss, epoch)

            main_pbar.update(1)

    elif params['func'] == 'evaluate':
        logger.info('Analyze FLOPs and Number of Parameters')
        # net.print_num_parameters()
        net.print_flops()

        logger.info('Evaluation started')
        pbar = tqdm(total=len(evalDS), position=0)
        pbar.set_description(f"[Evaluation][{params['checkpoint']}]")
        epoch_loss = evaluate(net, evalDL, params, pbar, eval_data_configs)
        pbar.close()

        for key in epoch_loss:
            loss = np.mean(np.array(epoch_loss[key]))
            logger.info(f"{key}-Loss --- mean: {np.mean(loss):.4f} std: {np.std(loss):.4f}")

        # xlsx_path = os.path.join(str(result_path), f"{params['output']}_reg_statistics.xlsx")
        # save_results_to_xlsx(epoch_loss, xlsx_path)
        # logger.info(f"Save statistics to {xlsx_path}")

if __name__ == '__main__':
    main()
