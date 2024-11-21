import os
import pickle

import matplotlib.pyplot as plt
import torch
from IPython import embed

from src.evals import eval_darkroom
from src.utils import (
    build_data_filename,
    build_model_filename,
)
import numpy as np
import scipy
import time
import hydra
import json
import glob
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['test'] = True  # TODO: won't work for context-length evaluation
    model_config['horizon'] = env_config['horizon']
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config

    # Resume wandb run from training
    model_dir_path = build_model_filename(
        env_config, model_config, optimizer_config)
    try:
        run_info_path = os.path.join(
            cfg.storage_dir, 'models', model_dir_path, 'run_info.json')
        with open(run_info_path, "r") as f:
            run_id = json.load(f)['run_id']
    except FileNotFoundError:
        raise ValueError("Could not locate wandb ID for trained model.")
    wandb_project = cfg.wandb.project
    wandb_logger = WandbLogger(
        project=wandb_project,
        id=run_id,  # Specify the run to resume
        resume="must"  # Must resume the exact run
    )

    # Instantiate model and load checkpoint  # TODO: Seed?
    model = instantiate(model_config)
    model = model.to(device)
    if cfg.epoch > 0:
        ckpt_pattern = f'{cfg.storage_dir}/models/{model_dir_path}/'
        ckpt_pattern += f'epoch={cfg.epoch}-val_loss=*.ckpt'
        matching_files = glob.glob(ckpt_pattern)
        if len(matching_files) > 0:
            ckpt_name = os.path.basename(matching_files[0])
        else:
            raise ValueError(f"No checkpoint found for epoch {cfg.epoch}")
    else:
        ckpt_name = 'last.ckpt'  # Use last checkpoint if epoch not specified
    checkpoint = torch.load(
        f'{cfg.storage_dir}/models/{model_dir_path}/{ckpt_name}')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load trajectories
    eval_traj_filepath = build_data_filename(
        env_config, mode=2, storage_dir=cfg.storage_dir + '/datasets')
    with open(eval_traj_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)
    n_eval = min(cfg.n_eval, len(eval_trajs))

    # Online and offline evaluation.
    if env_config['env'] == 'darkroom':
        H = cfg.H if cfg.H > 0 else env_config['horizon']
        config = {
            'Heps': 40,
            'horizon': env_config['horizon'],  # Horizon in an episode
            'H': H,  # Number of episodes to keep in context. TODO: not really used?
            'n_eval': min(20, n_eval),
            'dim': env_config['dim'],
        }
        eval_darkroom.online(eval_trajs, model, **config)
        fig = plt.gcf()
        wandb_logger.experiment.log({"online_performance": wandb.Image(fig)}) 
        plt.clf()

        del config['Heps']
        del config['H']
        config['n_eval'] = n_eval
        eval_darkroom.offline(eval_trajs, model, **config)
        fig = plt.gcf()
        wandb_logger.experiment.log({"offline_performance": wandb.Image(fig)}) 
        plt.clf()

if __name__ == '__main__':
    main()
