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
    set_all_seeds
)
import numpy as np
import seaborn as sns
import pandas as pd
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb import init, log
from copy import deepcopy
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
    train_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(0))
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
    with open(train_dset_path, 'rb') as f:  # TODO: switch to training dataset
        train_trajs = pickle.load(f)  # List of dicts
    n_eval_envs = min(cfg.n_eval_envs, len(train_trajs))
    train_trajs = train_trajs[:n_eval_envs]
    env_config['initialization_seed'] = [
        train_trajs[i_eval]['initialization_seed'] for i_eval in range(len(train_trajs))]

    # Fully offline evaluation of context length-dependency 
    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': []
    }
    context_lengths = np.linspace(100, max_context_length, 20, dtype=int)
    context_lengths = context_lengths[context_lengths < 1200]
    context_lengths_to_visualize = [
        context_lengths[context_lengths.size//10],
        context_lengths[context_lengths.size//3],
        context_lengths[2*(context_lengths.size//3)],
        context_lengths[-1]
        ]
    for context_length in context_lengths:
        log_and_visualize = context_length in context_lengths_to_visualize
        _results = eval_offline_by_context_length(
            model_config, env_config, optimizer_config, train_trajs,
            context_length, cfg.test_horizon, cfg.n_eval_episodes,
            log_and_visualize)
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


def eval_offline_by_context_length(
    model_config, env_config, optimizer_config, train_trajs,
    context_length, test_horizon, n_eval_episodes, log_and_visualize):
    print(f'\nEvaluating context length {context_length}')

    results = {
        'returns': [],
        'environment': [],
        'experienced_reward': [],
        'context_length': [],
    }
    eval_func = EvalTrees()
    agent_trajectories = [[], []]
    for i, traj in enumerate(train_trajs):
        print(f'...environment {i}')
        _traj = {}
        for k in traj.keys():
            val = traj[k][:context_length] if 'context' in k else traj[k]
            _traj[k] = val
        experienced_reward = np.sum(_traj['context_rewards']).item()
        env = eval_func.create_env(env_config, _traj['goal'], i)
        env.horizon = test_horizon

        set_all_seeds(i)
        model = instantiate(model_config)
        if model_config['name'] == 'dqn':
            model = model.to(device)
        _log_and_visualize = log_and_visualize and i==0 
        _returns, _trajectory = train_and_eval_agent(
            model, env, optimizer_config, _traj, n_eval_episodes,
            log_and_visualize=_log_and_visualize)
        set_all_seeds()
        results['returns'].append(_returns)
        results['environment'].append(i)
        results['experienced_reward'].append(experienced_reward)
        results['context_length'].append(context_length)
        if log_and_visualize and i < 3:
            agent_trajectories[0].append(_trajectory)
            agent_trajectories[1].append(env)

    if log_and_visualize:
        fig, ax = plt.subplots(figsize=(15, 3))
        eval_func.plot_trajectory(agent_trajectories[0], agent_trajectories[1], ax)
        wandb.log({"sample_paths_context_len_{}".format(context_length): wandb.Image(fig)}) 
        plt.clf()

    return results


def train_and_eval_agent(
        model, env, optimizer_config, traj, n_eval_episodes,
        log_and_visualize=False, debug=False):
    eval_func = EvalTrees()

    model.store_transition_from_dict(traj)
    n_training_samples = model.get_buffer_size()
    n_training_epochs = optimizer_config['num_epochs']
    eval_every = max(1, n_training_epochs // 25)
    eval_envs = [env.clone() for _ in range(n_eval_episodes)]

    if log_and_visualize:
        wandb.define_metric("custom_step")
        wandb.define_metric(f"training_loss_H{n_training_samples}", step_metric="custom_step")
        wandb.define_metric(f"test_returns_H{n_training_samples}", step_metric="custom_step")

    best_q_loss = float('inf')
    best_q_loss_epoch = None
    best_q_loss_state_dict = None
    for i in range(n_training_epochs):
        losses = model.training_epoch()
        loss = np.mean(losses)
        if log_and_visualize:
            wandb.log({f"training_loss_H{n_training_samples}": loss, "custom_step": i})
        if (i % eval_every == 0) or (i == n_training_epochs - 1):
            epoch_eval_returns, _ = model.deploy_vec(eval_envs, env.horizon)
            epoch_eval_returns = np.mean(epoch_eval_returns)

            if log_and_visualize:
                wandb.log({f"test_returns_H{n_training_samples}": epoch_eval_returns, "custom_step": i})
            if debug:
                epoch_eval_returns = []
                for _ in range(n_eval_episodes):
                    _epoch_returns, _trajectory = model.deploy(
                        env, horizon=env.horizon, debug=True)
                    epoch_eval_returns.append(_epoch_returns)
                epoch_eval_returns = np.mean(epoch_eval_returns)
                fig, ax = plt.subplots(figsize=(15, 3))
                eval_func.plot_trajectory([_trajectory]*3, [env]*3, ax)
                wandb.log({"debug_paths": wandb.Image(fig)}) 
                plt.clf()

        # Save the model checkpoint if it's the best so far
        if loss < best_q_loss:
            best_q_loss_state_dict = deepcopy(model.state_dict())
            best_q_loss_epoch = i
            best_q_loss = loss

    print_str = f"Evaluating {n_training_samples}-sample model loaded "
    print_str += f"from epoch {best_q_loss_epoch} with loss {best_q_loss}."
    print(print_str)
    model.load_state_dict(best_q_loss_state_dict)
    epoch_eval_returns, trajectories = model.deploy_vec(
        eval_envs, env.horizon)
    epoch_eval_returns = np.mean(epoch_eval_returns)

    return epoch_eval_returns, trajectories[0]


if __name__ == '__main__':
    main()
