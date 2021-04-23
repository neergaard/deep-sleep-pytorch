import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model.metrics as module_metric
from src.trainer.trainer import Trainer
from src.utils.config import process_config
from src.utils.factory import create_instance
from src.utils.logger import Logger


# Reproducibility
np.random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config, resume):
    train_logger = Logger()

    # Setup data_loader instances
    subsets = ['train', 'eval']
    datasets = {subset: create_instance(config.data_loader)(
        config, subset=subset) for subset in subsets}
    data_loaders = {subset: DataLoader(datasets[subset],
                                       batch_size=datasets[subset].batch_size,
                                       shuffle=True if subset is 'train' else False,
                                       num_workers=config.trainer.num_workers,
                                       drop_last=True,
                                       pin_memory=True) for subset in subsets}

    # build model architecture
    model = create_instance(config.network)(config)
    print(model)
    wandb.watch(model)

    # get function handles of loss and metrics
    loss = create_instance(config.loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = create_instance(config.optimizer)
    lr_scheduler = create_instance(config.lr_scheduler)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loaders['train'],
                      valid_data_loader=data_loaders['eval'],
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSleep')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # DEBUGGING:
    args.config = 'src/configs/exp03-frac100.yaml'
    # args.resume = 'experiments/exp01-hu2048/0502_122808/checkpoint-epoch39.pth'
    # args.device = '0'

    if args.config:
        # load config file
        config = process_config(args.config)

        # setting path to save trained models and log files
        path = os.path.join(config.trainer.save_dir, config.exp.name)

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.yaml', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    wandb.init(project='deep-sleep')
    wandb.config.update(config.toDict())

    main(config, args.resume)

    wandb.run.summary.update()
