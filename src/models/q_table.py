# src/models/dqn.py

import torch
import random
import numpy as np
from collections import defaultdict, deque
from src.utils import print_tree

class TabularQLearning:
    """Tabular Q-Learning class."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float,
        epsilon: float,
        optimizer_config: dict,
        name: str,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = optimizer_config['lr']
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.memory = deque()

    def get_buffer_size(self):
            return len(self.memory)

    def get_state_key(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if not isinstance(state, list):
            raise ValueError(f"Unexpected state type: {type(state)}")
        return tuple(state)

    def select_action(self, state, greedy=False):
        state_key = self.get_state_key(state)
        if random.random() > self.epsilon or greedy:
            return np.argmax(self.q_table[state_key])
        else:
            return random.randrange(self.action_dim)

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

    def deploy_vec(self, envs, horizon):
        return self.deploy(envs[0], horizon)

    def deploy(self, env, horizon, debug=False):
        state = env.reset()
        returns = 0
        trajectory = []

        for t in range(horizon):
            action = self.select_action(state, greedy=True)
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

        return returns, trajectory

    def store_transition_from_dict(self, traj_dict):
        n_samples, _ = traj_dict['context_states'].shape
        for i in range(n_samples):
            state = traj_dict['context_states'][i]
            action = traj_dict['context_actions'][i]
            reward = traj_dict['context_rewards'][i]
            next_state = traj_dict['context_next_states'][i]
            done = 0
            self.memory.append((state, action, reward, next_state, done))

    def shuffle_memory(self):
        temp_list = list(self.memory)
        random.shuffle(temp_list)
        self.memory = deque(temp_list)

    def training_epoch(self):
        losses = []
        for sample in self.memory:
            state, action, reward, next_state, done = sample
            action = np.argmax(action).item()
            td_error = self.update_q_table(state, action, reward, next_state, done)
            losses.append(td_error)
        self.shuffle_memory()
        return losses
