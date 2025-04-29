# src/models/dqn.py

import torch
import random
import numpy as np
from scipy.special import softmax
from collections import defaultdict, deque
from src.utils import print_tree

class TabularQLearning:
    """Tabular Q-Learning class, for a single environment! Doesn't have batching across environments."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float,
        action_temp: float,
        optimizer_config: dict,
        name: str,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.action_temp = action_temp
        self.alpha = optimizer_config['lr']
        self.q_table = defaultdict(self._default_q_values)
        self.memory = deque()

    def _default_q_values(self):
        """Return zero array of action_dim length"""
        return np.zeros(self.action_dim)

    def get_buffer_size(self):
            return len(self.memory)

    def select_action(self, state, greedy=False, action_temp=None):
        if action_temp is None:
            action_temp = self.action_temp
        state_key = self.get_state_key(state)
        q_values = self.q_table[state_key]
        probs = softmax(q_values/action_temp)
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action
    
    def select_action_vec(self, states, action_temp=None):
        if action_temp is None:
            action_temp = self.action_temp
        actions = [self.select_action(state, action_temp=action_temp) for state in states]
        return actions
    
    def store_transition_from_dict(self, traj_dict):
        n_samples, _ = traj_dict['context_states'].shape
        for i in range(n_samples):
            state = traj_dict['context_states'][i]
            action = traj_dict['context_actions'][i]
            reward = traj_dict['context_rewards'][i]
            next_state = traj_dict['context_next_states'][i]
            done = 0
            self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle_memory(self):
        temp_list = list(self.memory)
        random.shuffle(temp_list)
        self.memory = deque(temp_list)

    def get_state_key(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if not isinstance(state, list):
            raise ValueError(f"Unexpected state type: {type(state)}")
        return tuple(state)

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        best_next_action = np.argmax(self.q_table[next_state_key])
        if reward == 1:
            done = 1
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
        return td_error
    
    def debug_info(self, env):
        goal = env.goal.tolist()
        goal = env.node_map[tuple(goal)]
        print_tree(env)
        print(f'Goal: {goal}')

        readable_q_table = {}
        for key in self.q_table.keys():
            state_node = env.node_map[key]
            readable_q_table[(state_node.layer, state_node.pos)] = self.q_table[key]
        for key in readable_q_table.keys():
            print(f'{key}: {readable_q_table[key]}')

    def deploy(self, env, horizon, debug=False, max_normalize=False, action_temp=None):
        state = env.reset()
        dist_from_goal = env.dist_from_goal[tuple(state.tolist())]
        returns = 0
        trajectory = []

        for t in range(horizon):
            action = self.select_action(state, greedy=True, action_temp=action_temp)
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1
            next_state, reward, done, _ = env.step(action_vec)
            if debug:
                state_node = env.node_map[self.get_state_key(state)]
                next_state_node = env.node_map[self.get_state_key(next_state)]
                _str = f'({state_node.layer}, {state_node.pos}) with action {action}'
                _str += f'-> ({next_state_node.layer}, {next_state_node.pos}) with reward {reward}'
                print(_str)
            returns += reward
            trajectory.append(state)
            state = next_state
            if done:
                break
        if max_normalize:
            returns = returns / (horizon-dist_from_goal+1)
        return returns, trajectory

    def deploy_vec(self, envs, horizon, max_normalize=False, action_temp=None):
        num_envs = len(envs)
        returns = np.zeros(num_envs)
        trajectories = [[] for _ in range(num_envs)]
        states = [env.reset() for env in envs]
        dist_from_goals = [envs[i].dist_from_goal[tuple(states[i].tolist())] for i in range(num_envs)]
        
        for t in range(horizon):
            actions = self.select_action_vec(states, action_temp)
            for i, env in enumerate(envs):
                action_array = np.zeros(self.action_dim)
                action_array[actions[i]] = 1
                next_state, reward, done, _ = env.step(action_array)
                returns[i] += reward
                trajectories[i].append(states[i])
                states[i] = next_state
        if max_normalize:
            returns = [returns[i] / (horizon-dist_from_goals[i]+1) for i in range(num_envs)]
        return returns, trajectories

    def training_epoch(self):
        losses = []
        for sample in self.memory:
            state, action, reward, next_state, done = sample
            action = np.argmax(action).item()
            td_error = self.update_q_table(state, action, reward, next_state, done)
            losses.append(td_error)
        self.shuffle_memory()
        return losses

    def state_dict(self):
        return self.q_table
    
    def load_state_dict(self, state_dict):
        self.q_table = state_dict
