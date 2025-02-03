import argparse
import os
import pickle
import random

import gym
import numpy as np
from skimage.transform import resize
from IPython import embed
import hydra
from omegaconf import DictConfig, OmegaConf

from src.envs.darkroom_env import DarkroomEnv
from src.envs.maze_env import MazeEnv
from src.envs.trees import TreeEnv
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

    state = env.sample_state()
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

def generate_multiple_histories(env_class, env_configs, rollin_type, from_origin):
    """ Makes a list of trajectories from a list of environments. """
    trajs = []
    for env_config in env_configs:
        env = env_class(**env_config)
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = generate_history(env, rollin_type, from_origin)
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
        if env_class == TreeEnv:
            traj['initialization_seed'] = env.initialization_seed
        trajs.append(traj)
    return trajs


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

        env_configs = [{
            'dim': env_config['dim'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.repeat(train_goals, n_repeats, axis=0)]
        train_trajs = generate_multiple_histories(DarkroomEnv, env_configs, rollin_type, from_origin)
        print('Generated train trajectories.')

        env_configs = [{
            'dim': env_config['dim'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.repeat(test_goals, n_repeats, axis=0)]
        test_trajs = generate_multiple_histories(DarkroomEnv, env_configs, rollin_type, from_origin)
        print('Generated test trajectories.')

        env_configs = [{
            'dim': env_config['dim'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.repeat(eval_goals, n_repeats, axis=0)]
        eval_trajs = generate_multiple_histories(DarkroomEnv, env_configs, rollin_type, from_origin)
        print('Generated eval trajectories.')

    elif env_config['env'] == 'maze':
        layers = env_config['layers']
        goals = [(layers-1, p) for p in range(2**(layers-1))]
        n_envs = env_config['n_envs']
        n_repeats = n_envs // (len(goals))

        np.random.RandomState(seed=0).shuffle(goals)
        split_idx_1 = int(.8 * len(goals))
        split_idx_2 = int(.9 * len(goals))
        train_goals = goals[:split_idx_1]
        test_goals = goals[split_idx_1:split_idx_2]
        eval_goals = goals[split_idx_2:]

        env_configs = [{
            'layers': env_config['layers'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.tile(train_goals, (n_repeats, 1))]
        train_trajs = generate_multiple_histories(MazeEnv, env_configs, rollin_type, from_origin)
        print('Generated train trajectories.')

        env_configs = [{
            'layers': env_config['layers'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.tile(test_goals, (n_repeats, 1))]
        test_trajs = generate_multiple_histories(MazeEnv, env_configs, rollin_type, from_origin)
        print('Generated test trajectories.')

        env_configs = [{
            'layers': env_config['layers'], 'goal': goal, 'horizon': env_config['horizon']
            } for goal in np.tile(eval_goals, (n_repeats, 1))]
        eval_trajs = generate_multiple_histories(MazeEnv, env_configs, rollin_type, from_origin)
        print('Generated eval trajectories.')

    elif (env_config['env'] == 'tree') and (env_config['branching_prob'] == 1.):
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

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': i,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding'],
            'goal': goal
            } for i, goal in enumerate(np.tile(train_goals, (n_repeats, 1)))]
        train_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated train trajectories.')

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': i,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding'],
            'goal': goal
            } for i, goal in enumerate(np.tile(test_goals, (n_repeats, 1)))]
        test_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated test trajectories.')

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': i,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding'],
            'goal': goal
            } for i, goal in enumerate(np.tile(eval_goals, (n_repeats, 1)))]
        eval_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated eval trajectories.')

    elif (env_config['env'] == 'tree') and (env_config['branching_prob'] != 1.):
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

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': s,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding']
            } for s in np.tile(train_seeds, (n_repeats))]
        train_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated train trajectories.')

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': s,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding']
            } for s in np.tile(test_seeds, (n_repeats))]
        test_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated test trajectories.')

        env_configs = [{
            'max_layers': env_config['max_layers'],
            'initialization_seed': s,
            'horizon': env_config['horizon'],
            'branching_prob': env_config['branching_prob'],
            'node_encoding': env_config['node_encoding']
            } for s in np.tile(eval_seeds, (n_repeats))]
        eval_trajs = generate_multiple_histories(TreeEnv, env_configs, rollin_type, from_origin)
        print('Generated eval trajectories.')

    else:
        raise NotImplementedError
    

    train_filepath = os.path.join(dataset_storage_dir, build_dataset_name(0))
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
        print(f"Saved to {train_filepath}.")

    test_filepath = os.path.join(dataset_storage_dir, build_dataset_name(1))
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
        print(f"Saved to {test_filepath}.")

    eval_filepath = os.path.join(dataset_storage_dir, build_dataset_name(2))
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)
        print(f"Saved to {eval_filepath}.")


if __name__ == "__main__":
    main()
