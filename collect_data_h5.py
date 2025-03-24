import argparse
import os
import pickle
import random
import h5py

import gym
import numpy as np
from skimage.transform import resize
from IPython import embed
import hydra
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool

from src.envs.darkroom import DarkroomEnv
from src.envs.trees import TreeEnv
from src.envs.cntrees import CnTreeEnv
from src.utils import (
    build_env_name,
    build_dataset_name,
)


def generate_history(env, rollin_type, from_origin):
    """ Makes a trajectory from an environment. """
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.sample_state(from_origin=from_origin)
    if rollin_type == 'explore':
        env.update_exploration_buffer(None, state)
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        elif rollin_type == 'explore':
            action = env.explore_action()
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        if rollin_type == 'explore':
            env.update_exploration_buffer(action, next_state)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards


def process_single_env(args):
    """Helper function to process a single environment config for multiprocessing."""
    env_class, env_config, rollin_type, from_origin = args
    env = env_class(**env_config)
    context_states, context_actions, context_next_states, context_rewards = generate_history(
        env, rollin_type, from_origin
    )
    query_state = env.sample_state()
    optimal_action = env.opt_action(query_state)
    traj = {
        'query_state': query_state,
        'optimal_action': optimal_action,
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
        'goal': env.goal,
    }
    if isinstance(env, TreeEnv):
        traj['initialization_seed'] = env.initialization_seed
    return traj

def generate_multiple_histories(env_class, env_configs, rollin_type, from_origin):
    """Makes a list of trajectories from a list of environments using parallel processing."""
    args = [(env_class, env_config, rollin_type, from_origin) for env_config in env_configs]

    #with Pool(processes=n_processes) as pool: 
    with Pool() as pool:
        trajs = pool.map(process_single_env, args)
    
    return trajs

def package_env_configs(base_env_config, base_env_config_keys, new_config_vals, new_config_keys):
    """
    Packages a base environment configuration with new values for certain keys.
    """
    config_list = []
    n_configs = len(new_config_vals[0])
    for i in range(n_configs):
        config = {}
        for key in base_env_config_keys:
            config[key] = base_env_config[key]
        for key, val in zip(new_config_keys, new_config_vals):
            config[key] = val[i]
        config_list.append(config)
    return config_list

@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    np.random.seed(0)
    random.seed(0)
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_name = build_env_name(env_config)
    from_origin = cfg.from_origin
    rollin_type = env_config['rollin_type']
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    os.makedirs(dataset_storage_dir, exist_ok=True)

    if env_config['env'] == 'darkroom':
        goals = np.array([[(j, i) for i in range(env_config['dim'])]
                         for j in range(env_config['dim'])]).reshape(-1, 2)
        n_envs = env_config['n_envs']
        dim = env_config['dim']
        n_repeats = n_envs // (dim * dim)

        np.random.RandomState(seed=0).shuffle(goals)
        split_idx_1 = int(.8 * len(goals))
        split_idx_2 = int(.9 * len(goals))
        train_goals = goals[:split_idx_1]
        test_goals = goals[split_idx_1:split_idx_2]
        eval_goals = goals[split_idx_2:]

        train_env_configs = package_env_configs(
            env_config, ['dim', 'horizon'], [np.repeat(train_goals, n_repeats, axis=0)], ['goal'])
        test_env_configs = package_env_configs(
            env_config, ['dim', 'horizon'], [np.repeat(test_goals, n_repeats, axis=0)], ['goal'])
        eval_env_configs = package_env_configs(
            env_config, ['dim', 'horizon'], [np.repeat(eval_goals, n_repeats, axis=0)], ['goal'])
        EnvClass = DarkroomEnv

    elif ('tree' in env_config['env']) and (env_config['branching_prob'] == 1.):
        layers = env_config['max_layers']
        n_envs = env_config['n_envs']
        goals = [(layers-1, p) for p in range(2**(layers-1))]
        n_repeats = n_envs // (len(goals))

        np.random.RandomState(seed=0).shuffle(goals)
        split_idx_1 = int(.8 * len(goals))
        split_idx_2 = int(.9 * len(goals))
        train_goals = goals[:split_idx_1]
        test_goals = goals[split_idx_1:split_idx_2]
        eval_goals = goals[split_idx_2:]

        if env_config['env'] == 'tree':
            EnvClass = TreeEnv
            env_config_keys = [
                'max_layers', 'horizon',
                'branching_prob', 'node_encoding']
        elif env_config['env'] == 'cntree':
            EnvClass = CnTreeEnv
            env_config_keys = [
                'max_layers', 'horizon',
                'branching_prob', 'node_encoding_corr', 'state_dim']

        train_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(train_goals, (n_repeats, 1)), np.arange(n_repeats*len(train_goals))],
            ['goal', 'initialization_seed'])
        test_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(test_goals, (n_repeats, 1)), np.arange(n_repeats*len(test_goals))],
            ['goal', 'initialization_seed'])
        eval_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(eval_goals, (n_repeats, 1)), np.arange(n_repeats*len(eval_goals))],
            ['goal', 'initialization_seed'])

    elif ('tree' in env_config['env']) and (env_config['branching_prob'] != 1.):
        unique_seeds_path = dataset_storage_dir + '/unique_seeds.pkl'
        n_envs = env_config['n_envs']
        with open(unique_seeds_path, 'rb') as f:
            unique_seeds = pickle.load(f)
        train_seeds = unique_seeds['train']
        test_seeds = unique_seeds['test']
        eval_seeds = unique_seeds['eval']
        n_unique_seeds = len(train_seeds) + len(test_seeds) + len(eval_seeds)
        n_repeats = max(n_envs // n_unique_seeds, 1)
        print(f"n_repeats: {n_repeats}")

        if env_config['env'] == 'tree':
            EnvClass = TreeEnv
            env_config_keys = [
                'max_layers', 'horizon',
                'branching_prob', 'node_encoding']
        elif env_config['env'] == 'cntree':
            EnvClass = CnTreeEnv
            env_config_keys = [
                'max_layers', 'horizon',
                'branching_prob', 'node_encoding_corr', 'state_dim']

        train_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(train_seeds, (n_repeats))],
            ['initialization_seed'])
        test_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(test_seeds, (n_repeats))],
            ['initialization_seed'])
        eval_env_configs = package_env_configs(
            env_config, env_config_keys,
            [np.tile(eval_seeds, (n_repeats))],
            ['initialization_seed'])

    else:
        raise NotImplementedError
    
    train_trajs = generate_multiple_histories(EnvClass, train_env_configs, rollin_type, from_origin)
    print(f'Generated {len(train_trajs)} train trajectories.')
    train_filepath = os.path.join(dataset_storage_dir, 'train.h5')
    save_to_hdf5(train_filepath, train_trajs)
    print(f"Saved to {train_filepath}.")
    del train_trajs

    test_trajs = generate_multiple_histories(EnvClass, test_env_configs, rollin_type, from_origin)
    print(f'Generated {len(test_trajs)} test trajectories.')
    test_filepath = os.path.join(dataset_storage_dir, 'test.h5')
    save_to_hdf5(test_filepath, test_trajs)
    print(f"Saved to {test_filepath}.")
    del test_trajs  

    eval_trajs = generate_multiple_histories(EnvClass, eval_env_configs, rollin_type, from_origin)
    print(f'Generated {len(eval_trajs)} eval trajectories.')
    eval_filepath = os.path.join(dataset_storage_dir, 'eval.h5')
    save_to_hdf5(eval_filepath, eval_trajs)
    print(f"Saved to {eval_filepath}.")
    del eval_trajs


def save_to_hdf5(filepath, data):
    if os.path.exists(filepath):
        os.remove(filepath)
    with h5py.File(filepath, 'w') as h5file:
        for idx, entry in enumerate(data):
            group = h5file.create_group(str(idx))
            for key, value in entry.items():
                group.create_dataset(key, data=value)

if __name__ == "__main__":
    main()
