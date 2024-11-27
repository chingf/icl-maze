import os
import pickle

import matplotlib.pyplot as plt
import torch
from IPython import embed

from src.evals import eval_darkroom
from src.utils import (
    build_data_filename,
    build_model_filename,
    find_ckpt_file,
)
import numpy as np
import scipy
import time
import seaborn as sns
import pandas as pd
import hydra
import json
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
    ckpt_name = find_ckpt_file(cfg.storage_dir, model_dir_path, cfg.epoch)
    print(f'Loading checkpoint {ckpt_name}')
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
            'n_eval': n_eval,
            'dim': env_config['dim'],
        }
        #eval_darkroom.online(eval_trajs, model, **config)
        #fig = plt.gcf()
        #wandb_logger.experiment.log({"online_performance": wandb.Image(fig)}) 
        #plt.clf()

        #eval_darkroom.online(eval_trajs, model, random_buffer=True, **config)
        #fig = plt.gcf()
        #wandb_logger.experiment.log({"online_performance_random_buffer": wandb.Image(fig)}) 
        #plt.clf()

        #eval_darkroom.long_online(eval_trajs, model, **config)
        #fig = plt.gcf()
        #wandb_logger.experiment.log({"online_performance_long_context": wandb.Image(fig)}) 
        #plt.clf()

        #  TODO: Do this per context length, and show bar for the final context length
        del config['Heps']
        del config['H']
        config['n_eval'] = n_eval
        horizon = config['horizon']
        results = {
            'model': [],
            'return': [],
            'environment': [],
            'experienced_reward': [],
            'context_length': []
        }
        for _h in np.arange(0, horizon + 1, 2):

            # Generate truncated trajectories
            _eval_trajs = []
            for traj in eval_trajs:
                _traj = {}
                for k in traj.keys():
                    val = traj[k][:_h] if 'context' in k else traj[k]
                    _traj[k] = val
                _eval_trajs.append(_traj)
            experienced_rewards = [
                np.sum(_traj['context_rewards']).item() for _traj in _eval_trajs]

            # Evaluate offline
            _returns = eval_darkroom.offline(_eval_trajs, model, plot=True, **config)
            for model_name, model_returns in _returns.items():
                model_returns = model_returns.tolist()
                results['model'].extend([model_name] * n_eval)
                results['return'].extend(model_returns)
                results['environment'].extend([i for i in range(n_eval)])
                results['experienced_reward'].extend(experienced_rewards[:n_eval])
                results['context_length'].extend([_h]*n_eval)

            # Save performance when given full experience buffer
            if _h == horizon:
                fig = plt.gcf()
                wandb_logger.experiment.log(
                    {"offline_performance_horizon_{}".format(_h): wandb.Image(fig)}) 
            plt.clf()

        results = pd.DataFrame(results)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=results, x='context_length', y='return', hue='model',
            units='environment', estimator=None, ax=ax, alpha=0.5)
        plt.legend()
        wandb_logger.experiment.log({"offline_performance_horizon_comparison": wandb.Image(fig)}) 
        plt.clf()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=results, x='experienced_reward', y='return',
            hue='model', ax=ax)
        plt.legend()
        wandb_logger.experiment.log({"offline_performance_reward_comparison": wandb.Image(fig)}) 
        plt.clf()

if __name__ == '__main__':
    main()
