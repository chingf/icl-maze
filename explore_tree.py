import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.trees import TreeEnv


def generate_history(env, rollin_type):
    """ Makes a trajectory from an environment. """
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.sample_state()
    #state = env.reset(from_origin=True)
    if rollin_type == 'explore':
        env.update_exploration_buffer(None, state)
    for _ in range(env.horizon):
        if rollin_type == 'random':
            action = env.sample_action()
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

def generate_and_eval_multiple_histories(env_class, env_configs, rollin_type):
    """ Makes a list of trajectories from a list of environments. """
    trajs = []

    results = {
        'seed': [],
        'end_nodes_visited': [],
        'new_end_nodes_found': []
    }
    rewards = []
    for env_config in env_configs:
        env = env_class(**env_config)
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = generate_history(env, rollin_type)
        traj = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_next_states': context_next_states,
            'context_rewards': context_rewards,
        }
        end_nodes_visited, new_end_nodes_found = eval_traj_efficiency(env, traj)
        n_steps = len(end_nodes_visited)
        results['seed'].extend([env_config['initialization_seed']]*n_steps)
        results['end_nodes_visited'].extend(end_nodes_visited)
        results['new_end_nodes_found'].extend(new_end_nodes_found)
        rewards.append(np.sum(context_rewards))
    return results, rewards

def eval_traj_efficiency(env, traj):
    end_nodes_visited = [0]
    new_end_nodes_found = [0]
    seen_nodes = set()
    leaves = [l.encoding() for l in env.leaves]

    for state in traj['context_states']:
        state = tuple(state.tolist())
        if state not in leaves:
            continue

        if state in seen_nodes:
            end_nodes_visited.append(end_nodes_visited[-1]+1)
            new_end_nodes_found.append(new_end_nodes_found[-1])
        else:
            end_nodes_visited.append(end_nodes_visited[-1]+1)
            new_end_nodes_found.append(new_end_nodes_found[-1]+1)
            seen_nodes.add(state)

    return end_nodes_visited[1:], new_end_nodes_found[1:]

@hydra.main(version_base=None, config_path="configs/env", config_name="tree_explore")
def main(cfg: DictConfig):
    np.random.seed(0)
    random.seed(0)
    n_seeds = 30    

    # Test fully random rollouts
    env_configs = [{
        'max_layers': cfg.max_layers,
        'initialization_seed': s,
        'horizon': cfg.horizon,
        'branching_prob': cfg.branching_prob,
        } for s in range(n_seeds)]
    random_results, random_rewards = generate_and_eval_multiple_histories(TreeEnv, env_configs, rollin_type='random')
    random_results = pd.DataFrame(random_results)

    # Test exploration policy rollouts
    env_configs = [{
        'max_layers': cfg.max_layers,
        'initialization_seed': s,
        'horizon': cfg.horizon,
        'branching_prob': cfg.branching_prob,
        } for s in range(n_seeds)]
    explore_results, explore_rewards = generate_and_eval_multiple_histories(TreeEnv, env_configs, rollin_type='explore')
    explore_results = pd.DataFrame(explore_results)

    fig, ax = plt.subplots(figsize=(4,3))
    random_results['end_nodes_visited'] = np.log10(random_results['end_nodes_visited'])
    explore_results['end_nodes_visited'] = np.log10(explore_results['end_nodes_visited'])
    sns.lineplot(data=random_results, x='end_nodes_visited', y='new_end_nodes_found', label='Random', ax=ax)
    sns.lineplot(data=explore_results, x='end_nodes_visited', y='new_end_nodes_found', label='Explore', ax=ax)
    n_end_nodes = 2**(cfg.max_layers-1)
    ax.plot(
        np.log10(np.arange(1, n_end_nodes+1)), np.arange(1, n_end_nodes+1),
        color='black', label='Optimal')
    ax.set_xlabel('End nodes visited (log scale)')
    ax.set_ylabel('New end nodes found')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/explore_tree.png', dpi=300)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(random_rewards, label='Random', color='C0', alpha=0.5)
    ax.hist(explore_rewards, label='Explore', color='C1', alpha=0.5)
    ax.set_xlabel('Total reward exerienced')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/explore_tree_rewards.png', dpi=300)

if __name__ == "__main__":
    main()
