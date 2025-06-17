from copy import copy
import os
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
from src.agents.agent import TransformerAgent
from src.envs.cntrees import CnTreeEnv
from src.envs.trees import TreeEnv, TreeEnvVec
from src.envs.darkroom import DarkroomEnv, DarkroomEnvVec
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
    convert_to_tensor,
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

def create_env(config):
    if 'darkroom' in config['env']:
        _config = {
            'maze_dim': config['maze_dim'],
            'horizon': config['horizon'],
            'state_dim': config['state_dim'],
            'node_encoding_corr': config['node_encoding_corr'],
            'initialization_seed': config['initialization_seed'],
            'goal': config['goal']
        }
        return DarkroomEnv(**_config)
    elif 'tree' in config['env']:
        _config = {
            'max_layers': config['max_layers'],
            'horizon': config['horizon'],
            'branching_prob': config['branching_prob'],
            'goal': config['goal'],
            'initialization_seed': config['initialization_seed']
            }
        _config['node_encoding_corr'] = config['node_encoding_corr']
        _config['state_dim'] = config['state_dim']
        return CnTreeEnv(**_config)
    else:
        raise ValueError(f"Environment {config['env']} not supported.")

def offline(traj, model, config, env):
    """ Runs each episode separately with offline context. """

    n_eval_envs = config['n_eval_envs']
    n_eval_episodes = config['offline_eval_episodes']
    batch = {
        'context_states': convert_to_tensor([traj['context_states']]),
        'context_actions': convert_to_tensor([traj['context_actions']]),
        'context_next_states': convert_to_tensor([traj['context_next_states']]),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None]]),
        }

    # Load agents
    epsgreedy_agent = TransformerAgent(model, batch_size=1, sample=True)
    epsgreedy_agent_2 = TransformerAgent(model, batch_size=1, temp=1, sample=True)
    greedy_agent = TransformerAgent(model, batch_size=1, sample=False)
    epsgreedy_agent.set_batch(batch)
    epsgreedy_agent_2.set_batch(batch)
    greedy_agent.set_batch(batch)

    # Get unique states from context
    seen_states = np.vstack((traj['context_states'][:1], traj['context_next_states']))
    unique_states = np.unique(seen_states, axis=0)  # Get unique states
    possible_eval_states = []
    if 'darkroom' in config['env']:
        dist_threshold = 2
    elif 'tree' in config['env']:
        dist_threshold = env.max_layers-1
    else:
        raise ValueError(f"Environment {config['env']} not supported.")
    for state in unique_states:
        state_tuple = tuple(state.tolist())
        if env.dist_from_goal[state_tuple] >= dist_threshold:
            possible_eval_states.append(state_tuple)
    if len(possible_eval_states) == 0:
        baselines = {
            'Learner (temp=2)': 0,
            'Learner (temp=1)': 0,
            'Learner (greedy)': 0,
        }
        return baselines, None

    # Deploy agents offline
    env.reset_state_bank = possible_eval_states
    if isinstance(env, DarkroomEnv):
        vec_env = DarkroomEnvVec([env])
    else:
        vec_env = TreeEnvVec([env])
    greedy_returns = []
    epsgreedy_returns = []
    epsgreedy_returns_2 = []
    opt_epsgreedy_returns = []
    opt_epsgreedy_returns_2 = []
    opt_greedy_returns = []

    for _ in range(n_eval_episodes):
        _epsgreedy_obs, _, _, _epsgreedy_returns, _opt_returns_epsgreedy = \
            vec_env.deploy_eval(epsgreedy_agent, return_max_rewards=True)
        _epsgreedy_obs_2, _, _, _epsgreedy_returns_2, _opt_returns_epsgreedy_2 = \
            vec_env.deploy_eval(epsgreedy_agent_2, return_max_rewards=True)
        _greedy_obs, _, _, _greedy_returns, _opt_returns_greedy = \
            vec_env.deploy_eval(greedy_agent, return_max_rewards=True)
        epsgreedy_returns.append(np.sum(_epsgreedy_returns))
        epsgreedy_returns_2.append(np.sum(_epsgreedy_returns_2))
        greedy_returns.append(np.sum(_greedy_returns))
        opt_epsgreedy_returns.append(_opt_returns_epsgreedy[0])
        opt_epsgreedy_returns_2.append(_opt_returns_epsgreedy_2[0])
        opt_greedy_returns.append(_opt_returns_greedy[0])

    epsgreedy_returns = np.mean(np.array(epsgreedy_returns)/np.array(opt_epsgreedy_returns))
    epsgreedy_returns_2 = np.mean(np.array(epsgreedy_returns_2)/np.array(opt_epsgreedy_returns_2))
    greedy_returns = np.mean(np.array(greedy_returns)/np.array(opt_greedy_returns))

    print(f"Epsgreedy returns: {epsgreedy_returns}")
    print(f"Greedy returns: {greedy_returns}")
    print()

    # Plot and return
    baselines = {
        'Learner (temp=2)': epsgreedy_returns,
        'Learner (temp=1)': epsgreedy_returns_2,
        'Learner (greedy)': greedy_returns,
    }

    return baselines, _epsgreedy_obs

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
    if 'darkroom' in env_config['env']:
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'maze_dim': env_config['maze_dim'],
            'state_dim': env_config['state_dim'],
            'node_encoding_corr': env_config['node_encoding_corr'],
            'env': env_config['env']
        }
    elif env_config['env'] == 'tree':
        config = {
            'online_eval_episodes': cfg.online_eval_episodes,
            'online_eps_in_context': cfg.online_eps_in_context,
            'offline_eval_episodes': cfg.offline_eval_episodes,
            'horizon': cfg.test_horizon,
            'n_eval_envs': n_eval_envs,
            'max_layers': env_config['max_layers'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding'],
            'env': env_config['env']
        }
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
            'env': env_config['env']
        }
    else:
        raise ValueError(f"Environment {env_config['env']} not supported.")


    # Offline evaluations
    results = {
        'model': [],
        'return': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': []
    }
    if 'darkroom' in env_config['env']:
        context_lengths = np.arange(0, 250, 10)
    elif 'tree' in env_config['env']:
        context_lengths = np.concatenate([
            np.arange(0, 325, 25),  # 0 to 300 in steps of 25
            np.arange(400, max_context_length + 100, 100)  # 400 onwards in steps of 100
        ])
    else:
        raise ValueError(f"Environment {env_config['env']} not supported.")
    for traj_idx, traj in enumerate(eval_trajs):
        print('Environment: ', traj_idx)
        env_config = copy(config)
        env_config['initialization_seed'] = traj['initialization_seed']
        env_config['goal'] = traj['goal']
        env = create_env(env_config)
        env.optimal_action_map, env.dist_from_goal = env.make_opt_action_dict()
        for context_length in context_lengths:
            print('Context length: ', context_length)
            truncated_traj = {}
            for k in traj.keys():
                if 'context' in k:
                    val = traj[k][:context_length]
                elif k == 'initialization_seed':
                    val = np.array(traj[k]).item()
                elif k == 'goal':
                    val = np.array(traj[k])
                else:
                    val = traj[k]
                truncated_traj[k] = val
            experienced_rewards = np.sum(truncated_traj['context_rewards']).item()
            returns, obs = offline(truncated_traj, model, env_config, env)
            for model_name, model_returns in returns.items():
                results['model'].append(model_name)
                results['return'].append(model_returns)
                results['environment'].append(traj_idx)
                results['experienced_reward'].append(experienced_rewards)
                results['context_length'].append(context_length)

    results = pd.DataFrame(results)
    results['path_length_scaled'] = 1 - results['return']

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
        data=results, x='context_length', y='return', hue='model',
        units='environment', estimator=None, ax=ax, alpha=0.2)
    sns.lineplot(
        data=results, x='context_length', y='return', hue='model',
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

    with open(os.path.join(model_storage_dir, 'eval_results_offline.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
