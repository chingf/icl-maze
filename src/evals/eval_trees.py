import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.agents.agent import (
    TransformerAgent,
    OptPolicy,
)
from src.envs.trees import (
    TreeEnv,
    TreeEnvVec,
)
from src.utils import convert_to_tensor
from src.evals.eval_darkroom import EvalDarkroom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalTrees(EvalDarkroom):
    def __init__(self):
        pass

    def create_env(self, config, goal, i_eval):
        _config = {
            'max_layers': config['max_layers'],
            'horizon': config['horizon'],
            'branching_prob': config['branching_prob'],
            'node_encoding': config['node_encoding'],
            'goal': goal
            }
        if isinstance(config['initialization_seed'], list):
            _config['initialization_seed'] = config['initialization_seed'][i_eval]
        else:
            _config['initialization_seed'] = config['initialization_seed']
        return TreeEnv(**_config)

    def create_vec_env(self, envs):
        return TreeEnvVec(envs)
    
    def online(self, eval_trajs, model, config):
        config['initialization_seed'] = [
            eval_trajs[i_eval]['initialization_seed'] for i_eval in range(len(eval_trajs))]
        return super().online(eval_trajs, model, config)

    def offline(self, eval_trajs, model, config, return_envs=False):
        """Runs each episode separately with offline context, after calculating tree-specific metrics."""

        config['initialization_seed'] = [
            eval_trajs[i_eval]['initialization_seed'] for i_eval in range(len(eval_trajs))]
        config['goals'] = [
            eval_trajs[i_eval]['goal'] for i_eval in range(len(eval_trajs))]
        return super().offline(eval_trajs, model, config, return_envs)

    def continual_online(self, eval_trajs, model, config):
        config['initialization_seed'] = [
            eval_trajs[i_eval]['initialization_seed'] for i_eval in range(len(eval_trajs))]
        return super().continual_online(eval_trajs, model, config)
    
    def plot_trajectory(self, obs, envs, ax):
        per_path_offset = 2**(envs[0].max_layers+3)
        max_nodes_per_layer = 2**(envs[0].max_layers)
        for i in range(3):  # Arbitrarily plot 3 sample paths
            path = []  # Collect path taken by agent
            for observation in obs[i]:
                env = envs[i]
                if not isinstance(observation, list):
                    observation = observation.tolist()
                node = env.node_map[tuple(observation)]
                path.append([node.layer, node.pos])
            path = np.array(path).astype(float)
            base = 2*np.ones(path.shape[0])
            nodes_per_layer = np.power(base, path[:,0])
            offset_chunks = (4*max_nodes_per_layer)/(nodes_per_layer+1)
            path[:,1] += offset_chunks*(path[:,1]+1)
            path[:,1] += i*per_path_offset
            path[:,0] += np.random.normal(0, 0.1, path[:,0].shape)  # y-jitter
            path[:,1] += np.random.normal(0, 1, path[:,1].shape)  # x-jitter

            maze = []  # Collect all nodes in the tree
            for node in env.node_map.values():
                maze.append([node.layer, node.pos])
            maze = np.array(maze).astype(float)
            base = 2*np.ones(maze.shape[0])
            nodes_per_layer = np.power(base, maze[:,0])
            offset_chunks = (4*max_nodes_per_layer)/(nodes_per_layer+1)   
            maze[:,1] += offset_chunks*(maze[:,1]+1)
            maze[:,1] += i*per_path_offset  # Spaces out paths from different seeds

            # Plot actual tree
            ax.scatter(
                maze[:,1], -maze[:,0],
                c='gray', alpha=0.2, s=30, marker='x')

            # Plot path
            scatter = ax.scatter(
                path[:, 1], 
                -path[:, 0],
                c=np.arange(len(path)), 
                cmap='viridis',
                alpha=0.4,
                s=5
            )
            
        # Add colorbar to show temporal progression
        plt.colorbar(scatter, label='Timestep')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Agent Path - Environment {i}')