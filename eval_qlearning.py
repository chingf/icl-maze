import os
import pickle
import h5py
import random
import matplotlib.pyplot as plt
import torch
from src.envs.cntrees import CnTreeEnv
from src.envs.trees import TreeEnv
from src.envs.darkroom import DarkroomEnv, DarkroomEnvVec
from src.evals.eval_trees import EvalTrees
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
    set_all_seeds
)
import numpy as np
import seaborn as sns
import pandas as pd
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from copy import copy, deepcopy
wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
action_temps = [10.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

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
        if 'node_encoding_corr' in config:
            _config['node_encoding_corr'] = config['node_encoding_corr']
            _config['state_dim'] = config['state_dim']
            return CnTreeEnv(**_config)
        else:
            _config['node_encoding'] = config['node_encoding']
            return TreeEnv(**_config)
    else:
        raise ValueError(f"Environment {config['env']} not supported.")


@hydra.main(version_base=None, config_path="configs", config_name="eval_dqn")
def main(cfg: DictConfig):
    wandb_project = cfg.wandb.project
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config

    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    os.makedirs(model_storage_dir, exist_ok=True)
    if cfg.override_eval_dataset_path != None:
        eval_dset_path = cfg.override_eval_dataset_path
        for param in cfg.override_params:
            for k, v in param.items():
                env_config[k] = v
    else:
        eval_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(2))
    max_context_length = env_config['horizon']
    wandb_config = {
        'env': env_config,
        'model': model_config,
        'optimizer': optimizer_config,
        'n_eval_envs': cfg.n_eval_envs,
        'n_eval_episodes': cfg.n_eval_episodes,
        'test_horizon': cfg.test_horizon,
    }
    wandb.init(
        project=wandb_project,
        name=env_name + '/' + model_name,
        config=wandb_config,
        dir=cfg.storage_dir
    )

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
    env_config['initialization_seed'] = [
        np.array(eval_trajs[i_eval]['initialization_seed']).item() for i_eval in range(len(eval_trajs))]
    max_context_length = eval_trajs[0]['context_rewards'].shape[0]
    max_context_length = min(max_context_length, 1000)
    print(f'Max context length: {max_context_length}')

    # Fully offline evaluation of context length-dependency 
    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': [],
        'action_temps': [],
    }
    if 'darkroom' in env_config['env']:
        context_lengths = np.arange(10, 250, 10)
    elif 'tree' in env_config['env']:
        context_lengths = np.concatenate([
            np.arange(0, 325, 25),  # 0 to 300 in steps of 25
            np.arange(400, max_context_length + 100, 100)  # 400 onwards in steps of 100
        ])
        context_lengths[0] = 10
    else:
        raise ValueError(f"Environment {env_config['env']} not supported.")
    context_lengths_to_visualize = [
        context_lengths[context_lengths.size//10],
        context_lengths[context_lengths.size//3],
        context_lengths[2*(context_lengths.size//3)],
        context_lengths[-1],
    ]

    for i, traj in enumerate(eval_trajs):
        print(f'...environment {i}')
        _results = eval_env(
            model_config, env_config, optimizer_config,
            traj, i,
            cfg.test_horizon, cfg.n_eval_episodes, cfg.n_eval_envs, cfg.continual_weights,
            context_lengths, context_lengths_to_visualize,
            model_storage_dir)
        results = {k: results[k] + _results[k] for k in results.keys()}


    ## How does performance vary with context length?
    results = pd.DataFrame(results)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='context_length', y='returns',
        units='environment', estimator=None, ax=ax, alpha=0.2)
    sns.lineplot(
        data=results, x='context_length', y='returns',
        ax=ax)
    wandb.log({"offline_performance_context_length_comparison": wandb.Image(fig)}) 
    plt.clf()

    ## How does performance vary with experienced reward?
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='experienced_reward', y='returns',
        ax=ax)
    wandb.log({"offline_performance_reward_comparison": wandb.Image(fig)}) 
    plt.clf()

    with open(os.path.join(model_storage_dir, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def eval_env(
    model_config, env_config, optimizer_config,
    traj, env_idx,
    test_horizon, n_eval_episodes, n_eval_envs, continual_weights,
    context_lengths, context_lengths_to_visualize,
    model_storage_dir
    ):

    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': [],
        'action_temps': [],
    }
    env_config = copy(env_config)
    env_config['initialization_seed'] = traj['initialization_seed']
    env_config['goal'] = traj['goal']
    env = create_env(env_config)
    env.optimal_action_map, env.dist_from_goal = env.make_opt_action_dict()
    env.horizon = test_horizon

    set_all_seeds(env_idx)
    model = instantiate(model_config)
    if model_config['name'] == 'dqn':
        model = model.to(device)
    set_all_seeds()

    for context_length in context_lengths:
        print(f'\nEvaluating context length {context_length}')
        _traj = {}
        for k in traj.keys():
            if 'context' in k:
                val = traj[k][:context_length]
            elif k == 'initialization_seed':
                val = np.array(traj[k]).item()
            elif k == 'goal':
                val = np.array(traj[k])
            else:  # optimal_action and query_state shouldn't be needed in eval
                val = traj[k]
            _traj[k] = val
        experienced_reward = np.sum(_traj['context_rewards']).item()
        log_and_visualize = context_length in context_lengths_to_visualize
        _returns, _trajectories, _action_temps, _state_dict = train_and_eval_agent(
            model, env, optimizer_config, _traj, n_eval_episodes,
            log_and_visualize=log_and_visualize and (env_idx == 0))
        results['returns'].extend(_returns)
        results['action_temps'].extend(_action_temps)
        results['environment'].extend([env_idx] * len(_returns))
        results['experienced_reward'].extend([experienced_reward] * len(_returns))
        results['context_length'].extend([context_length] * len(_returns))

        if continual_weights and _state_dict is not None:
            model.load_state_dict(_state_dict)
        else:
            set_all_seeds(env_idx)
            model = instantiate(model_config)
            if model_config['name'] == 'dqn':
                model = model.to(device)
            set_all_seeds()

    with open(os.path.join(model_storage_dir, f'traj_{env_idx}_state_dict.pkl'), 'wb') as f:
        pickle.dump(_state_dict, f)

    return results


def train_and_eval_agent(
        model, env, optimizer_config, traj, n_eval_episodes,
        log_and_visualize=False, debug=False):
    model.store_transition_from_dict(traj)
    n_training_samples = model.get_buffer_size()
    n_training_epochs = optimizer_config['num_epochs']
    eval_every = max(1, n_training_epochs // 5)

    seen_states = np.vstack((traj['context_states'][:1], traj['context_next_states']))
    unique_states = np.unique(seen_states, axis=0)  # Get unique states
    possible_eval_states = []
    for state in unique_states:
        state_tuple = tuple(state.tolist())
        if isinstance(env, DarkroomEnv):
            dist_threshold = 2
        elif isinstance(env, CnTreeEnv):
            dist_threshold = env.max_layers-1
        else:
            raise ValueError(f"Environment not supported.")
        if env.dist_from_goal[state_tuple] >= dist_threshold:
            possible_eval_states.append(state_tuple)
    experienced_reward = np.sum(traj['context_rewards']).item()
    print("rewards experienced: ", experienced_reward)
    print("Number of seen states: ", unique_states.shape[0])

    if len(possible_eval_states) == 0:
        return 0, None, None
    if log_and_visualize:
        wandb.define_metric("custom_step")
        wandb.define_metric(f"training_loss_H{n_training_samples}", step_metric="custom_step")
        wandb.define_metric(f"test_returns_H{n_training_samples}", step_metric="custom_step")

    best_q_loss = float('inf')
    best_q_loss_epoch = None
    best_q_loss_state_dict = None
    eval_envs = [env.clone() for _ in range(n_eval_episodes)]
    for i in range(n_training_epochs):
        losses = model.training_epoch()
        loss = np.mean(losses)
        if log_and_visualize:
            wandb.log({f"training_loss_H{n_training_samples}": loss, "custom_step": i})
        if (i % eval_every == 0) or (i == n_training_epochs - 1):
            epoch_eval_returns, _ = model.deploy_vec(eval_envs, env.horizon, max_normalize=True)
            epoch_eval_returns = np.mean(epoch_eval_returns)

            if log_and_visualize:
                wandb.log({f"test_returns_H{n_training_samples}": epoch_eval_returns, "custom_step": i})
            if debug:
                epoch_eval_returns = []
                for _ in range(n_eval_episodes):
                    _epoch_returns, _trajectory = model.deploy(
                        env, horizon=env.horizon, debug=True, max_normalize=True)
                    epoch_eval_returns.append(_epoch_returns)
                epoch_eval_returns = np.mean(epoch_eval_returns)

        # Save the model checkpoint if it's the best so far
        if loss < best_q_loss:
            best_q_loss_state_dict = deepcopy(model.state_dict())
            best_q_loss_epoch = i
            best_q_loss = loss

    print_str = f"Evaluating {n_training_samples}-sample model loaded "
    print_str += f"from epoch {best_q_loss_epoch} with loss {best_q_loss}."
    print(print_str)
    model.load_state_dict(best_q_loss_state_dict)
    returns = []
    trajectories = []
    for action_temp in action_temps:
        print(action_temp)
        _returns, _trajectories = model.deploy_vec(
            eval_envs, env.horizon, max_normalize=True, action_temp=action_temp)
        returns.append(np.mean(_returns))
        trajectories.append(_trajectories[0])

    print("Eval returns: ", returns)

    return returns, trajectories, action_temps, best_q_loss_state_dict


if __name__ == '__main__':
    main()
