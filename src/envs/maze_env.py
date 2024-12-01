import itertools
import gym
import numpy as np
import torch
from src.envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MazeEnv(BaseEnv):
    def __init__(self, layers, goal, horizon):
        self.layers = layers
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 4  # (back, left, right, stay)

    def sample_state(self):
        layer = np.random.randint(0, self.layers)
        pos = np.random.randint(0, 2**layer)
        return np.array([layer, pos])

    def sample_action(self):
        i = np.random.randint(0, self.action_dim)
        a = np.zeros(self.action_dim)
        a[i] = 1
        return a

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_dim)
        layer, pos = state
        if action == 0:  # Back
            if layer > 0:
                new_layer = layer - 1
                new_pos = pos//2
                state = [new_layer, new_pos]
        elif action == 1:  # Left
            if layer < self.layers-1:
                new_layer = layer + 1
                new_pos = 2*pos
                state = [new_layer, new_pos]
        elif action == 2:  # Right
            if layer < self.layers-1:
                new_layer = layer + 1
                new_pos = 2*pos + 1
                state = [new_layer, new_pos]
        elif action == 3:  # Stay
            pass

        if np.all(state == self.goal):
            reward = 1
        else:
            reward = 0
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()
    
    def opt_action(self, state):
        if state[0] > self.goal[0]:
            action = 0
        elif state[0] < self.goal[0]:
            layer_diff = self.goal[0] - state[0]
            left_tree_start_node = (2**(layer_diff-1))*(2*state[1])
            right_tree_start_node = (2**(layer_diff-1))*(2*state[1]+1)
            left_tree_end_node = left_tree_start_node + (2**(layer_diff-1)-1)
            right_tree_end_node = right_tree_start_node + (2**(layer_diff-1)-1)
            goal_node = self.goal[1]
            if goal_node >= left_tree_start_node and goal_node <= left_tree_end_node:
                action = 1  # Left
            elif goal_node >= right_tree_start_node and goal_node <= right_tree_end_node:
                action = 2  # Right
            else:
                action = 0  # Backtrack
        elif state[0] == self.goal[0]:
            if state[1] == self.goal[1]:
                action = 3 # Stay
            else:
                action = 0 #  Backtrack
        zeros = np.zeros(self.action_dim)
        zeros[action] = 1
        return zeros


class MazeEnvVec(BaseEnv):
    """
    Vectorized Maze environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews