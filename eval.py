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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['test'] = True
    model_config['horizon'] = env_config['horizon']
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']

    # Seed management
    tmp_seed = seed = cfg['seed']  # TODO: figure out what's going on with seeds
    if seed == -1:
        tmp_seed = 0
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

    # Instantiate model and load checkpoint
    model_dir_path = build_model_filename(env_config, model_config)
    model = instantiate(model_config)
    model = model.to(device)
    epoch_name = f'epoch{cfg.epoch}' if cfg.epoch > 0 else 'final'
    checkpoint = f'pickles/models/{model_dir_path}/{epoch_name}.pt'
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load trajectories
    eval_traj_filepath = build_data_filename(env_config, mode=2)
    with open(eval_traj_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)
    n_eval = min(cfg.n_eval, len(eval_trajs))

    # Manage directories for plots
    eval_fig_dir_path = f'figs/{model_dir_path}/eval/{epoch_name}'
    os.makedirs(eval_fig_dir_path, exist_ok=True)
    os.makedirs(f'{eval_fig_dir_path}/bar', exist_ok=True)
    os.makedirs(f'{eval_fig_dir_path}/online', exist_ok=True)
    os.makedirs(f'{eval_fig_dir_path}/graph', exist_ok=True)

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
        plt.savefig(f'{eval_fig_dir_path}/online.png')
        plt.clf()

        del config['Heps']
        del config['H']
        config['n_eval'] = n_eval
        eval_darkroom.offline(eval_trajs, model, **config)
        plt.savefig(f'{eval_fig_dir_path}/bar.png')
        plt.clf()

if __name__ == '__main__':
    main()
