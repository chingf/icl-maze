import os
import pickle
import matplotlib.pyplot as plt
import torch
from IPython import embed
from src.evals.eval_trees import EvalTrees
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
)
import numpy as np
import seaborn as sns
import pandas as pd
import hydra
import json
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb import init, log
wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="configs", config_name="eval_dqn")
def main(cfg: DictConfig):
    wandb_project = cfg.wandb.project
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config
    if env_config['env'] != 'tree':
        raise ValueError(f"Environment {env_config['env']} not supported")

    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    eval_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(2))
    horizon = env_config['horizon']
    wandb.init(project=wandb_project, dir=cfg.storage_dir)

    # Load trajectories
    with open(eval_dset_path, 'rb') as f:
        eval_trajs = pickle.load(f)  # List of dicts
    n_eval = min(cfg.n_eval, len(eval_trajs))
    eval_trajs = [eval_trajs[8]]

    env_config['initialization_seed'] = [
        eval_trajs[i_eval]['initialization_seed'] for i_eval in range(len(eval_trajs))]

    # Fully offline evaluation of context length-dependency 
    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': []
    }
    test_horizons = np.linspace(0, horizon, 50, dtype=int)
    test_horizons[0] = 10
    horizons_to_visualize = [
        test_horizons[test_horizons.size//10],
        test_horizons[test_horizons.size//3],
        test_horizons[2*(test_horizons.size//3)],
        test_horizons[-1]
        ]
    for test_horizon in [horizon]: #test_horizons:
        visualize_trajectory = horizon in horizons_to_visualize
        _results = eval_offline_by_context_length(
            model_config, env_config, optimizer_config, eval_trajs,
            test_horizon, visualize_trajectory)
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
    plt.legend()
    wandb.log({"offline_performance_horizon_comparison": wandb.Image(fig)}) 
    plt.clf()

    ## How does performance vary with experienced reward?
    fig, ax = plt.subplots()
    sns.lineplot(
        data=results, x='experienced_reward', y='returns',
        ax=ax)
    plt.legend()
    wandb.log({"offline_performance_reward_comparison": wandb.Image(fig)}) 
    plt.clf()


def eval_offline_by_context_length(
    model_config, env_config, optimizer_config, eval_trajs,
    horizon, visualize_trajectory):

    # Generate truncated trajectories
    config = {
        'horizon': env_config['horizon'],  # Horizon in an episode
        'max_layers': env_config['max_layers'],
        'branching_prob': env_config['branching_prob'],
        'node_encoding': env_config['node_encoding']
    }

    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': [],
    }
    eval_func = EvalTrees()
    agent_trajectories = [[], []]
    for i, traj in enumerate(eval_trajs):
        _traj = {}
        for k in traj.keys():
            val = traj[k][:horizon] if 'context' in k else traj[k]
            _traj[k] = val
        experienced_reward = np.sum(_traj['context_rewards']).item()
        model = instantiate(model_config)
        if model_config['name'] == 'dqn':
            model = model.to(device)
        env = eval_func.create_env(env_config, _traj['goal'], i)
        _returns, _trajectory = train_and_eval_agent(model, env, optimizer_config, _traj)
        results['returns'].append(_returns)
        results['environment'].append(i)
        results['experienced_reward'].append(experienced_reward)
        results['context_length'].append(horizon)
        if visualize_trajectory and i < 3:
            agent_trajectories[0].append(_trajectory)
            agent_trajectories[1].append(env)

    if visualize_trajectory:
        fig, ax = plt.subplots(figsize=(15, 3))
        eval_func.plot_trajectory(agent_trajectories[0], agent_trajectories[1], ax)
        wandb.log({"sample_paths_context_len_{}".format(horizon): wandb.Image(fig)}) 
        plt.clf()

    return results


def train_and_eval_agent(model, env, optimizer_config, traj):
    eval_func = EvalTrees()

    model.store_transition_from_dict(traj)
    n_data_samples = model.get_buffer_size()
    trajectory = None
    n_training_epochs = optimizer_config['num_epochs']
    n_eval_episodes = 1  # TODO: change to 10
    wandb.define_metric("custom_step")
    wandb.define_metric(f"training_loss_H{n_data_samples}", step_metric="custom_step")
    wandb.define_metric(f"test_returns_H{n_data_samples}", step_metric="custom_step")
    for i in range(n_training_epochs):
        losses = model.training_epoch()
        loss = np.mean(losses)
        wandb.log({f"training_loss_H{n_data_samples}": loss, "custom_step": i})
        if (i % 5 == 0) or (i == n_training_epochs - 1):
            epoch_eval_returns = []
            for _ in range(n_eval_episodes):
                debug = False if i < 1000 else True 
                _epoch_returns, _trajectory = model.deploy(env, horizon=env.horizon, debug=debug)
                epoch_eval_returns.append(_epoch_returns)
            epoch_eval_returns = np.mean(epoch_eval_returns)
            wandb.log({f"test_returns_H{n_data_samples}": epoch_eval_returns, "custom_step": i})
            if (i % 100==0):  # TODO: debugging
                fig, ax = plt.subplots(figsize=(15, 3))
                t = [_trajectory]*3
                e = [env]*3
                eval_func.plot_trajectory(t, e, ax)
                wandb.log({"sample_paths_context_len_{}".format(n_data_samples): wandb.Image(fig)}) 
                plt.clf()

            # TODO: Save model
    return epoch_eval_returns, trajectory


if __name__ == '__main__':
    main()
