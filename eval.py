import os
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
from IPython import embed

from src.evals.eval_darkroom import EvalDarkroom
from src.evals.eval_trees import EvalTrees, EvalCntrees
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
    find_ckpt_file,
)
import numpy as np
import random
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

    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    if cfg.override_eval_dataset_path != None:
        eval_dset_path = cfg.override_eval_dataset_path
        for param in cfg.override_params:
            for k, v in param.items():
                env_config[k] = v
    else:
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

    # Instantiate model and load checkpoint
    model = instantiate(model_config)
    model = model.to(device)
    ckpt_name = find_ckpt_file(model_storage_dir, cfg.epoch)
    print(f'Loading checkpoint {ckpt_name}')
    checkpoint = torch.load(os.path.join(model_storage_dir, ckpt_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load trajectories
    is_h5_file = eval_dset_path.endswith('.h5')
    if is_h5_file:
        eval_trajs = h5py.File(eval_dset_path, 'r')
        traj_indices = list(eval_trajs.keys())
        n_eval_envs = min(cfg.n_eval_envs, len(traj_indices))
        random.seed(0)
        traj_indices = random.sample(traj_indices, n_eval_envs)
        random.seed()
        eval_trajs = [eval_trajs[i] for i in traj_indices]
    else:  # Pickle file
        with open(eval_dset_path, 'rb') as f:
            eval_trajs = pickle.load(f)
        n_eval_envs = min(cfg.n_eval_envs, len(eval_trajs))
        random.seed(0)
        eval_trajs = random.sample(eval_trajs, n_eval_envs)
        random.seed()
    max_context_length = eval_trajs[0]['context_rewards'].shape[0]
    max_context_length = min(max_context_length, 1000)
    print(f'Max context length: {max_context_length}')

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
    elif env_config['env'] == 'cntree':
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'max_layers': env_config['max_layers'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding_corr': env_config['node_encoding_corr'],
            'state_dim': env_config['state_dim'],
        }
        eval_func = EvalCntrees()


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
    context_lengths = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600, 700, 800, 900, 1000])
    context_lengths_to_visualize = [
        context_lengths[context_lengths.size//10],
        context_lengths[context_lengths.size//3],
        context_lengths[2*(context_lengths.size//3)],
        context_lengths[-1],
        ]
    for _context_length in context_lengths:  # TODO: revert
        _eval_trajs = []
        for traj in eval_trajs:  # Generate truncated trajectories
            _traj = {}
            for k in traj.keys():
                if 'context' in k:
                    val = traj[k][:_context_length]
                elif k == 'initialization_seed':
                    val = np.array(traj[k]).item()
                elif k == 'goal':
                    val = np.array(traj[k])
                else:  # optimal_action and query_state shouldn't be needed in eval
                    val = traj[k]
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

        if _context_length in context_lengths_to_visualize and 'tree' in env_config['env']:
            if _obs is not None:
                fig, ax = plt.subplots(figsize=(15, 3))
                eval_func.plot_trajectory(_obs, _envs, ax)
                wandb_logger.experiment.log(
                    {"sample_paths_context_len_{}".format(_context_length): wandb.Image(fig)}) 
                plt.clf()

    results = pd.DataFrame(results)
    opt_return = results[results['model']=='Opt']['return'].mean()
    results['path_length_scaled'] = (opt_return - results['return'])/opt_return
    results['returns_scaled'] = results['return']/opt_return

    ## Save performance given the full context length
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
        data=results, x='context_length', y='returns_scaled', hue='model',
        units='environment', estimator=None, ax=ax, alpha=0.2)
    sns.lineplot(
        data=results, x='context_length', y='returns_scaled', hue='model',
        ax=ax)
    plt.legend()
    wandb_logger.experiment.log(
        {"offline_returns_v_clen": wandb.Image(fig)}) 
    plt.clf()

    ## How does performance vary with experienced reward?
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='experienced_reward', y='path_length_scaled',
        hue='model', ax=ax)
    plt.legend()
    wandb_logger.experiment.log({"offline_pathlen_v_expreward": wandb.Image(fig)}) 
    plt.clf()

    results = results[results['model'] != 'Opt']
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=results, x='experienced_reward', y='path_length_scaled',
        hue='model', ax=ax)
    # Add exponential fit lines for each model
    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]
        x = model_data['experienced_reward']
        y = model_data['path_length_scaled']
        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        try:
            popt, _ = curve_fit(exp_func, x, y, p0=[1, 1e-3, 0])
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = exp_func(x_fit, *popt)
            #color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(x_fit, y_fit, '--', label=f'{model_name} (fit)')
        except:
            continue
    plt.legend()
    wandb_logger.experiment.log({"offline_pathlen_v_expreward_scatter": wandb.Image(fig)}) 
    plt.clf()

    with open(os.path.join(model_storage_dir, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
