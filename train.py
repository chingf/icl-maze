import json
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
wandb.login()

import numpy as np
import random
from src.dataset import Dataset, ImageDataset
from src.utils import (
    build_data_filename,
    build_model_filename,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="configs", config_name="training")
def main(cfg: DictConfig):
    np.random.seed(0)  # TODO: figure out what's happening with seeds
    random.seed(0)
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['test'] = False
    model_config['horizon'] = env_config['horizon']
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config
    model = instantiate(model_config)
    model = model.to(device)

    # Random seed handling 
    tmp_seed = seed = cfg.seed  # TODO: figure out what's happening with seeds
    if seed == -1:
        tmp_seed = 0
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    # Set up directories
    path_train = build_data_filename(
        env_config, mode=0, storage_dir=cfg.storage_dir + '/datasets')
    path_test = build_data_filename(
        env_config, mode=1, storage_dir=cfg.storage_dir + '/datasets')
    model_chkpt_path = build_model_filename(
        env_config, model_config, optimizer_config)
    os.makedirs(f'{cfg.storage_dir}/models/{model_chkpt_path}', exist_ok=True)

    # Set up datasets and dataloaders
    train_dataset = Dataset(path_train, env_config)
    test_dataset = Dataset(path_test, env_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, optimizer_config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, optimizer_config['batch_size'])
    
    # Set up logging and checkpointing
    wandb_config = {
        'env': env_config,
        'model': model_config,
        'optimizer': optimizer_config,
        'seed': seed
    }
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=model_chkpt_path,
        config=wandb_config,
        save_dir='logs'
    )

    # Save run ID for later evaluation
    run_info = {
        "run_id": wandb_logger.experiment.id,
        "run_name": wandb_logger.experiment.name,
        "version": wandb_logger.version  # This is the same as run_id
    }
    with open(f'{cfg.storage_dir}/models/{model_chkpt_path}/run_info.json', 'w') as f:
        json.dump(run_info, f)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg.storage_dir}/models/{model_chkpt_path}',
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=optimizer_config['num_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='auto',  # TODO: maybe need to specify devices manually
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )

if __name__ == "__main__":
    main()