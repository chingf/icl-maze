# src/models/dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from itertools import islice
from src.utils import print_tree


class DQN(nn.Module):
    """Deep Q-Network class."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        gamma: float,
        epsilon: float,
        target_update: int,
        optimizer_config: dict,
        name: str,
        buffer_size: int=None,
    ):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.target_update = target_update
        self.memory = deque(maxlen=buffer_size)
        self.training_epochs_done = 0

        self.q_network = self.make_mlp()
        self.target_network = self.make_mlp()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = self.make_optimizer(optimizer_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_mlp(self):
        layers = []
        in_dim = self.state_dim
        for layer in range(self.n_layers):
            if layer == 0:
                hidden_dim = 256
            else:
                hidden_dim = max(in_dim//2, 16)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        return nn.Sequential(*layers)

    def get_buffer_size(self):
        return len(self.memory)

    def forward(self, state):
        return self.q_network(state)

    def select_action(self, state, greedy=False):
        sample = random.random() if not greedy else 1
        if sample > self.epsilon:
            with torch.no_grad():
                return self.q_network(state).argmax().view(1, 1)
        else:
            random_action = random.randrange(self.action_dim)
            return torch.tensor([[random_action]], device=self.device, dtype=torch.int64)

    def store_transition_from_dict(self, traj_dict):
        n_samples, _ = traj_dict['context_states'].shape
        for i in range(n_samples):
            state = traj_dict['context_states'][i]
            action = traj_dict['context_actions'][i]
            reward = traj_dict['context_rewards'][i]
            next_state = traj_dict['context_next_states'][i]
            done = 0 if reward == 0 else 1
            self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle_memory(self):
        temp_list = list(self.memory)
        random.shuffle(temp_list)
        self.memory = deque(temp_list)

    def update_target_network(self):
        if self.training_epochs_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def get_state_key(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if not isinstance(state, list):
            raise ValueError(f"Unexpected state type: {type(state)}")
        return tuple(state)

    def make_q_table(self, env):
        q_table = {}
        for state in env.node_map.keys():
            state_vec = torch.tensor(state, device=self.device, dtype=torch.float32)
            q_table[state] = self.q_network(state_vec).detach().cpu().numpy()
        return q_table

    def debug_info(self, env):
        goal = env.goal.tolist()
        goal = env.node_map[tuple(goal)]
        q_table = self.make_q_table(env)
        print_tree(env)
        print(f'Goal: {goal}')

        readable_q_table = {}
        for key in q_table.keys():
            state_node = env.node_map[key]
            readable_q_table[(state_node.layer, state_node.pos)] = q_table[key]
        for key in readable_q_table.keys():
            print(f'{key}: {readable_q_table[key]}')

    def deploy(self, env, horizon, debug=False):
        if debug:
            import pdb; pdb.set_trace()
        with torch.no_grad():
            state = torch.tensor(env.reset(), device=self.device, dtype=torch.float32)
            returns = 0
            trajectory = []
            
            for t in range(horizon):
                action = self.select_action(state, greedy=True)
                action_array = np.zeros(self.action_dim)
                action_array[action.item()] = 1
                next_state, reward, done, _ = env.step(action_array)
                if debug:
                    state_node = env.node_map[self.get_state_key(state)]
                    next_state_node = env.node_map[self.get_state_key(next_state)]
                    _str = f'({state_node.layer}, {state_node.pos}) with action {action.item()}'
                    _str += f'-> ({next_state_node.layer}, {next_state_node.pos}) with reward {reward}'
                    print(_str)
                returns += reward
                trajectory.append(state.cpu().numpy())
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
            return returns, trajectory

    ## Optimization functions below

    def make_optimizer(self, optimizer_config: dict):
        optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
        self.batch_size = optimizer_config['batch_size']
        return optimizer

    def training_epoch(self):
        n_transitions = len(self.memory)
        losses = []
        
        for i in range(0, n_transitions, self.batch_size):
            end_idx = min(i + self.batch_size, n_transitions)
            batch = islice(self.memory, i, end_idx)
            loss = self.optimization_step(batch)
            losses.append(loss)
        self.update_target_network()
        self.shuffle_memory()
        return losses

    def optimization_step(self, batch):  # TODO: double check Q loss
        batch = list(zip(*batch))

        state_batch = torch.tensor(
            np.stack(batch[0]), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(
            np.argmax(np.stack(batch[1]), axis=1),
            device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(
            np.stack(batch[2]), device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(
            np.stack(batch[3]), device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(
            np.stack(batch[4]), device=self.device, dtype=torch.float32)

        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        next_state_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        max_grad_norm = 0.5  # You can adjust this value
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_grad_norm)
        self.optimizer.step()

        return loss.item()
