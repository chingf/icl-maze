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
from src.utils import (
    build_env_name,
    build_dataset_name,
)


def generate_history(env, rollin_type):
    """ Makes a trajectory from an environment. """
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.sample_state()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

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

def generate_histories_from_envs(envs, env_config):
    """ Makes a list of trajectories from a list of environments. """
    trajs = []
    for env in envs:  # TODO: can double up with n_hists and n_samples
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = generate_history(env, env_config['rollin_type'])
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
        trajs.append(traj)
    return trajs

def generate_darkroom_histories(goals, env_config):
    """ Creates darkroom environments and then makes trajectories. """
    envs = [DarkroomEnv(
        env_config['dim'], goal, env_config['horizon']) for goal in goals]
    trajs = generate_histories_from_envs(envs, env_config)
    return trajs

def generate_maze_histories(goals, env_config):
    """ Creates darkroom environments and then makes trajectories. """
    envs = [MazeEnv(
        env_config['layers'], goal, env_config['horizon']) for goal in goals]
    trajs = generate_histories_from_envs(envs, env_config)
    return trajs


@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    np.random.seed(0)
    random.seed(0)
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_name = build_env_name(env_config)
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
        eval_goals = np.repeat(eval_goals, n_repeats, axis=0)
        train_goals = np.repeat(train_goals, n_repeats, axis=0)
        test_goals = np.repeat(test_goals, n_repeats, axis=0)

        train_trajs = generate_darkroom_histories(train_goals, env_config)
        test_trajs = generate_darkroom_histories(test_goals, env_config)
        eval_trajs = generate_darkroom_histories(eval_goals, env_config)

        train_filepath = os.path.join(dataset_storage_dir, build_dataset_name(0))
        test_filepath = os.path.join(dataset_storage_dir, build_dataset_name(1))
        eval_filepath = os.path.join(dataset_storage_dir, build_dataset_name(2))
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
        train_goals = np.tile(train_goals, (n_repeats, 1))
        test_goals = np.tile(test_goals, (n_repeats, 1))
        eval_goals = np.tile(eval_goals, (n_repeats, 1))

        train_trajs = generate_maze_histories(train_goals, env_config)
        test_trajs = generate_maze_histories(test_goals, env_config)
        eval_trajs = generate_maze_histories(eval_goals, env_config)

        train_filepath = os.path.join(dataset_storage_dir, build_dataset_name(0))
        test_filepath = os.path.join(dataset_storage_dir, build_dataset_name(1))
        eval_filepath = os.path.join(dataset_storage_dir, build_dataset_name(2))
    else:
        raise NotImplementedError

    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
        print(f"Saved to {train_filepath}.")
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
        print(f"Saved to {test_filepath}.")
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)
        print(f"Saved to {eval_filepath}.")


if __name__ == "__main__":
    main()
