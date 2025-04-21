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
from src.evals.eval_darkroom import EvalDarkroom
from src.evals.eval_trees import EvalTrees, EvalCntrees
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
    _config = {
        'max_layers': config['max_layers'],
        'horizon': config['horizon'],
        'branching_prob': config['branching_prob'],
        'goal': config['goal'],
        'initialization_seed': config['initialization_seed']
        }
    if 'node_encoding_corr' in config:
        _config['node_encoding_corr'] = config['node_encoding_corr']
        _config['state_dim'] = config['state_dim']
        return CnTreeEnv(**_config)
    else:
        _config['node_encoding'] = config['node_encoding']
        return TreeEnv(**_config)

def offline(traj, model, config, env, query_type):
    """ Runs each episode separately with offline context. """

    n_eval_envs = config['n_eval_envs']
    n_eval_episodes = config['offline_eval_episodes']
    batch = {
        'context_states': convert_to_tensor([traj['context_states']]),
        'context_actions': convert_to_tensor([traj['context_actions']]),
        'context_next_states': convert_to_tensor([traj['context_next_states']]),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None]]),
        }
    rewards = np.argwhere(np.array(traj['context_rewards'])>0)
    first_reward = rewards.squeeze()[0]
    last_reward = rewards.squeeze()[-1]
    # Load agents
    epsgreedy_agent = TransformerAgent(model, batch_size=1, sample=True)
    epsgreedy_agent_2 = TransformerAgent(model, batch_size=1, temp=1, sample=True)
    greedy_agent = TransformerAgent(model, batch_size=1, sample=False)
    epsgreedy_agent.set_batch(batch)
    epsgreedy_agent_2.set_batch(batch)
    greedy_agent.set_batch(batch)

    # Get unique states from context
    all_states = np.array([list(k) for k in env.node_map.keys()])
    seen_states_pre_reward = np.vstack((traj['context_states'][:1], traj['context_next_states'][:first_reward]))
    seen_states_post_reward = np.vstack((traj['context_states'][last_reward:last_reward+1], traj['context_next_states'][last_reward:]))
    seen_states_pre_reward = np.unique(seen_states_pre_reward, axis=0)  # Get unique states
    seen_states_post_reward = np.unique(seen_states_post_reward, axis=0)  # Get unique states
    seen_states_post_reward = np.array([state for state in seen_states_post_reward 
        if not any(np.array_equal(state, pre_state) 
        for pre_state in seen_states_pre_reward)])
    all_seen_states = np.vstack((seen_states_pre_reward, seen_states_post_reward))
    unseen_states = np.array([state for state in all_states
        if not any(np.array_equal(state, seen_state) 
        for seen_state in all_seen_states)])
    
    if query_type == 'seen_states_pre_reward':
        query_state_bank = seen_states_pre_reward
    elif query_type == 'seen_states_post_reward':
        query_state_bank = seen_states_post_reward
    elif query_type == 'unseen_states':
        query_state_bank = unseen_states
    else:
        raise ValueError("Invalid query type.")

    possible_eval_states = []
    for state in query_state_bank:
        state_tuple = tuple(state.tolist())
        if env.dist_from_goal[state_tuple] >= env.max_layers-1:
            possible_eval_states.append(state_tuple)
    if len(possible_eval_states) == 0:
        return None, None

    # Deploy agents offline
    env.reset_state_bank = possible_eval_states
    vec_env = TreeEnvVec([env])
    greedy_returns = []
    epsgreedy_returns = []
    epsgreedy_returns_2 = []
    opt_returns = []

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
        opt_returns.append(_opt_returns_epsgreedy)
    epsgreedy_returns = np.mean(epsgreedy_returns)
    epsgreedy_returns_2 = np.mean(epsgreedy_returns_2)
    greedy_returns = np.mean(greedy_returns)
    opt_returns = np.mean(opt_returns)

    print(f"Epsgreedy returns: {epsgreedy_returns}")
    print(f"Greedy returns: {greedy_returns}")
    print()

    # Plot and return
    baselines = {
        'Opt': opt_returns,
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

    assert env_config['env'] == 'cntree'
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

    # Offline evaluations
    results = {
        'model': [],
        'return': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': [],
        'query_type': []
    }
    for traj_idx, traj in enumerate(eval_trajs):
        print('Environment: ', traj_idx)
        env_config = copy(config)
        env_config['initialization_seed'] = traj['initialization_seed']
        env_config['goal'] = traj['goal']
        env = create_env(env_config)
        env.optimal_action_map, env.dist_from_goal = env.make_opt_action_dict()
        truncated_traj = {}
        for k in traj.keys():
            if 'context' in k:
                val = traj[k][:max_context_length]
            elif k == 'initialization_seed':
                val = np.array(traj[k]).item()
            elif k == 'goal':
                val = np.array(traj[k])
            else:
                val = traj[k]
            truncated_traj[k] = val
        first_reward = np.argwhere(np.array(truncated_traj['context_rewards'])>0)
        if first_reward.size == 0:
            continue
        first_reward = first_reward.squeeze()[0]
        if first_reward > max_context_length-200:
            continue
        experienced_rewards = np.sum(truncated_traj['context_rewards']).item()

        for query_type in ['seen_states_pre_reward', 'seen_states_post_reward', 'unseen_states']:
            returns, obs = offline(truncated_traj, model, env_config, env, query_type)
            if returns is None:
                continue
            for model_name, model_returns in returns.items():
                results['model'].append(model_name)
                results['return'].append(model_returns)
                results['environment'].append(traj_idx)
                results['experienced_reward'].append(experienced_rewards)
                results['context_length'].append(max_context_length)
                results['query_type'].append(query_type)

    results = pd.DataFrame(results)
    opt_return = results[results['model']=='Opt']['return'].mean()
    results['path_length_scaled'] = (opt_return - results['return'])/opt_return
    results['returns_scaled'] = results['return']/opt_return

    with open(os.path.join(model_storage_dir, 'eval_results_offline_by_query_type.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
