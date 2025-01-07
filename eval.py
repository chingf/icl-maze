import os
import pickle
import matplotlib.pyplot as plt
import torch
from IPython import embed

from src.evals.eval_darkroom import EvalDarkroom
from src.evals.eval_maze import EvalMaze
from src.evals.eval_trees import EvalTrees
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
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
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config
    wandb_project = cfg.wandb.project
    max_context_length = env_config['horizon']

    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    eval_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(2))

    # Resume wandb run from training
    try:
        run_info_path = os.path.join(model_storage_dir, 'run_info.json')
        with open(run_info_path, "r") as f:
            run_id = json.load(f)['run_id']
    except FileNotFoundError:
        raise ValueError("Could not locate wandb ID for trained model.")
    wandb_logger = WandbLogger(
        project=wandb_project,
        id=run_id,  # Specify the run to resume
        resume="must",  # Must resume the exact run
        save_dir=cfg.storage_dir
    )

    # Instantiate model and load checkpoint  # TODO: Seed?
    model = instantiate(model_config)
    model = model.to(device)
    ckpt_name = find_ckpt_file(model_storage_dir, cfg.epoch)
    print(f'Loading checkpoint {ckpt_name}')
    checkpoint = torch.load(os.path.join(model_storage_dir, ckpt_name))
    # Below used for debugging
    #mpath = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/icl-maze/"
    #mpath += "7layer/tree_layers7_bprob0.9_envs300000_H800_explore/models/" 
    #mpath += "transformer_end_query_embd512_layer4_head4_lr0.0001_drop0.1_batch256/copied_best.ckpt"
    #checkpoint = torch.load(mpath)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load trajectories
    with open(eval_dset_path, 'rb') as f:
        eval_trajs = pickle.load(f)
    n_eval_envs = min(cfg.n_eval_envs, len(eval_trajs))
    eval_trajs = eval_trajs[:n_eval_envs]

    # Online and offline evaluation.
    if env_config['env'] == 'darkroom':
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'dim': env_config['dim'],
        }
        eval_func = EvalDarkroom()
    elif env_config['env'] == 'maze':
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'layers': env_config['layers'],
        }
        eval_func = EvalMaze()
    elif env_config['env'] == 'tree':
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'max_layers': env_config['max_layers'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding']
        }
        eval_func = EvalTrees()

#    eval_func.continual_online(eval_trajs, model, config)
#    fig = plt.gcf()
#    wandb_logger.experiment.log({"continual_online_performance": wandb.Image(fig)}) 
#    plt.clf()
#
#    eval_func.online(eval_trajs, model, config)
#    fig = plt.gcf()
#    wandb_logger.experiment.log({"online_performance": wandb.Image(fig)}) 
#    plt.clf()

    # Offline evaluations
    results = {
        'model': [],
        'return': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': []
    }
    context_lengths = np.linspace(0, max_context_length, 20, dtype=int)
    context_lengths_to_visualize = [
        context_lengths[context_lengths.size//10],
        context_lengths[context_lengths.size//3],
        context_lengths[2*(context_lengths.size//3)],
        context_lengths[-1]
        ]
    for _context_length in [800]: #context_lengths:
        _eval_trajs = []
        for traj in eval_trajs:  # Generate truncated trajectories
            _traj = {}
            for k in traj.keys():
                val = traj[k][:_context_length] if 'context' in k else traj[k]
                _traj[k] = val
            _eval_trajs.append(_traj)
        experienced_rewards = [
            np.sum(_traj['context_rewards']).item() for _traj in _eval_trajs]

        _returns, _obs, _envs = eval_func.offline(  # Run offline
            _eval_trajs, model, config, return_envs=True)
        for model_name, model_returns in _returns.items():
            model_returns = model_returns.tolist()
            results['model'].extend([model_name] * n_eval_envs)
            results['return'].extend(model_returns)
            results['environment'].extend([i for i in range(n_eval_envs)])
            results['experienced_reward'].extend(experienced_rewards[:n_eval_envs])
            results['context_length'].extend([_context_length]*n_eval_envs)

        if _context_length in context_lengths_to_visualize and env_config['env'] == 'tree':
            fig, ax = plt.subplots(figsize=(15, 3))
            eval_func.plot_trajectory(_obs, _envs, ax)
            wandb_logger.experiment.log(
                {"sample_paths_context_len_{}".format(_context_length): wandb.Image(fig)}) 
            plt.clf()

    ## Save performance given the full context length
    results = pd.DataFrame(results)
    max_ctxt_length_results = results[results['context_length'] == max_context_length]
    fig, ax = plt.subplots()
    sns.barplot(data=max_ctxt_length_results, x='model', y='return', ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel('Average Return')
    plt.title(f'Performance with Maximum Context Length ({max_context_length})')
    wandb_logger.experiment.log(
        {"offline_performance_context_length_{}".format(max_context_length): wandb.Image(fig)})
    plt.clf()

    ## How does performance vary with context length?
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='context_length', y='return', hue='model',
        units='environment', estimator=None, ax=ax, alpha=0.2)
    sns.lineplot(
        data=results, x='context_length', y='return', hue='model',
        ax=ax)
    plt.legend()
    wandb_logger.experiment.log(
        {"offline_performance_context_length_comparison": wandb.Image(fig)}) 
    plt.clf()

    ## How does performance vary with experienced reward?
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='experienced_reward', y='return',
        hue='model', ax=ax)
    plt.legend()
    wandb_logger.experiment.log({"offline_performance_reward_comparison": wandb.Image(fig)}) 
    plt.clf()

if __name__ == '__main__':
    main()
