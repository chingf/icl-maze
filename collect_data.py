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
from src.utils import build_data_filename


def generate_history(env, rollin_type):
    """ Makes a trajectory from an environment. """
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
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


@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    np.random.seed(0)
    random.seed(0)
    env_config = OmegaConf.to_container(cfg.env, resolve=True)

    if env_config['env'] == 'darkroom':
        goals = np.array([[(j, i) for i in range(env_config['dim'])]
                         for j in range(env_config['dim'])]).reshape(-1, 2)
        n_envs = env_config['n_envs']
        dim = env_config['dim']

        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        train_goals = goals[:train_test_split]
        test_goals = goals[train_test_split:]
        eval_goals = np.array(test_goals.tolist() * int(100 // len(test_goals)))
        train_goals = np.repeat(train_goals, n_envs // (dim * dim), axis=0)
        test_goals = np.repeat(test_goals, n_envs // (dim * dim), axis=0)

        train_trajs = generate_darkroom_histories(train_goals, env_config)
        test_trajs = generate_darkroom_histories(test_goals, env_config)
        eval_trajs = generate_darkroom_histories(eval_goals, env_config)

        train_filepath = build_data_filename(
            env_config, mode=0, storage_dir=cfg.storage_dir + '/datasets')
        test_filepath = build_data_filename(
            env_config, mode=1, storage_dir=cfg.storage_dir + '/datasets')
        eval_filepath = build_data_filename(
            env_config, mode=2, storage_dir=cfg.storage_dir + '/datasets')
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
