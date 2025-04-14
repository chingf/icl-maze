import itertools
import gym
import numpy as np
import torch
import os
import pickle
from src.envs.base_env import BaseEnv

abs_path = '/n/home04/cfang/Code/icl-maze/src/envs/'  # so hacky

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DarkroomEnv(BaseEnv):
    def __init__(
        self, maze_dim, horizon, state_dim, node_encoding_corr,
        initialization_seed=None, goal=None
        ):
        self.maze_dim = maze_dim
        self.horizon = horizon
        self.initialization_seed = initialization_seed
        self.node_encoding_corr = node_encoding_corr
        self.action_dim = 5  # up, right, down, left, stay
        self.state_dim = state_dim
        if self.state_dim == 2:
            print('WARNING: State dim is 2, so defaulting to fixed (x, y) encoding')

        if initialization_seed is not None:
            np.random.seed(initialization_seed)
        self.node_map_encoding_to_pos = {}
        self.node_map_pos_to_encoding = {}
        self.maze = np.ones((maze_dim, maze_dim))
        self._generate_random_maze()
        if goal is None:
            self.goal = self.sample_goal()
        else:
            if isinstance(goal, np.ndarray) and goal.size == self.state_dim:
                self.goal = goal
            else:
                if isinstance(goal, np.ndarray):
                    goal = goal.tolist()
                if isinstance(goal, list):
                    goal = tuple(goal)
                self.goal = np.array(self.node_map_pos_to_encoding[goal])
        np.random.seed()
        
        self.optimal_action_map = None
        self.dist_from_goal = None
        self.exploration_buffer = {  # Only used when generating exploratory traj.
            'actions_made_in_curr_state': set(),
            'previous_state': None,
            'current_state': None,
            'last_action': None,
            }
        self.reset_state_bank = None

    def clone(self):
        env = DarkroomEnv(
            maze_dim=self.maze_dim,
            horizon=self.horizon,
            state_dim=self.state_dim,
            node_encoding_corr=self.node_encoding_corr,
            initialization_seed=self.initialization_seed,
            goal=self.goal
            )
        env.reset_state_bank = self.reset_state_bank
        if self.optimal_action_map is not None:
            env.optimal_action_map = self.optimal_action_map
        if self.dist_from_goal is not None:
            env.dist_from_goal = self.dist_from_goal
        return env

    def _sample_node_encoding(self, x, y, expansion_mat, dist_mat):
        loc = x * self.maze_dim + y
        encoding_vector = expansion_mat @ dist_mat[loc]
        encoding_vector = encoding_vector / np.linalg.norm(encoding_vector)
        encoding_vector = encoding_vector.astype(np.float32)
        return tuple(encoding_vector.tolist())

    def _generate_random_maze(self):
        expansion_mat = np.random.randn(
            self.state_dim, self.maze_dim**2).astype(np.float32)
        dist_mat = pickle.load(open(
            os.path.join(abs_path, f'maze_euclidean_dist_matrix_dim{self.maze_dim}.pkl'), 'rb'
            ))
        dist_mat = dist_mat.astype(np.float32)
        dist_mat = np.power(self.node_encoding_corr, dist_mat)

        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                if self.maze[x, y] == 0:
                    continue
                node_encoding = self._sample_node_encoding(x, y, expansion_mat, dist_mat)
                if self.state_dim == 2:
                    node_encoding = (x, y)
                self.node_map_encoding_to_pos[node_encoding] = (x, y)
                self.node_map_pos_to_encoding[(x, y)] = node_encoding

    def sample_goal(self):
        goal_pos = np.argwhere(self.maze == 1)
        goal_pos = goal_pos[np.random.choice(len(goal_pos))]
        goal_pos = tuple(goal_pos.tolist())
        goal_encoding = self.node_map_pos_to_encoding[goal_pos]
        return np.array(goal_encoding)

    def sample_state(self, from_origin=False):
        if self.reset_state_bank is not None:
            state = self.reset_state_bank[np.random.choice(len(self.reset_state_bank))]
            return np.array(state)
        else:
            valid_states = list(self.node_map_encoding_to_pos.keys())
            state = valid_states[np.random.choice(len(valid_states))]
            return np.array(state)

    def sample_action(self):
        i = np.random.randint(0, self.action_dim)
        a = np.zeros(self.action_dim)
        a[i] = 1
        return a

    def reset(self):
        self.optimal_action_map, self.dist_from_goal = self.make_opt_action_dict()
        self.current_step = 0

        attempts = 0
        while True:
            self.state = self.sample_state()
            if self.reset_state_bank is None:
                if self.dist_from_goal[tuple(self.state.tolist())] >= (self.maze_dim - 1):
                    break
            else:
                break
            attempts += 1
            if attempts > 200:
                raise ValueError("Failed to sample a valid state")
        return self.state
    
    def make_opt_action_dict(self):
        target_node_encoding = tuple(self.goal.tolist())
        return self._make_opt_action_dict(target_node_encoding)

    def _make_opt_action_dict(self, target_node_encoding):  # TODO
        target_x, target_y = self.node_map_encoding_to_pos[target_node_encoding]
        target_node_idx = target_x * self.maze_dim + target_y
        dist_mat = pickle.load(open(
            os.path.join(abs_path, f'maze_geodesic_dist_matrix_dim{self.maze_dim}.pkl'), 'rb'
            )) 
        opt_action_map = {}
        dist_from_goal = {}
        for node_encoding in self.node_map_encoding_to_pos.keys():
            node_x, node_y = self.node_map_encoding_to_pos[node_encoding]
            node_idx = node_x * self.maze_dim + node_y
            dist_from_goal[node_encoding] = dist_mat[node_idx, target_node_idx]
            opt_actions = self._get_opt_actions(node_x, node_y, target_x, target_y)
            opt_action_map[node_encoding] = np.random.choice(opt_actions)
        return opt_action_map, dist_from_goal

    def _get_opt_actions(self, node_x, node_y, target_x, target_y):
        if node_x == target_x:
            if node_y == target_y:  # At target
                opt_actions = [4]
            elif node_y < target_y:  # Target is to the right
                opt_actions = [1]
            else:  # Target is to the left
                opt_actions = [3]
        elif node_y == target_y:
            if node_x < target_x:  # Target is below
                opt_actions = [2]
            else:  # Target is above
                opt_actions = [0]
        else:
            if (node_x < target_x) and (node_y < target_y):  # Target is lower-right
                opt_actions = [2, 1]
            elif (node_x < target_x) and (node_y > target_y):  # Target is lower-left
                opt_actions = [2, 3]
            elif (node_x > target_x) and (node_y < target_y):  # Target is upper-right
                opt_actions = [0, 1]
            elif (node_x > target_x) and (node_y > target_y):  # Target is upper-left
                opt_actions = [0, 3]
        return opt_actions

    def opt_action(self, state):
        if self.optimal_action_map is None:
            self.optimal_action_map, self.dist_from_goal = self.make_opt_action_dict()

        if not isinstance(state, list):
            state = state.tolist()
        action = self.optimal_action_map[tuple(state)] 
        zeros = np.zeros(self.action_dim)
        zeros[action] = 1
        return zeros

    def explore_action(self):
        """Returns a random action"""
        current_state = self.exploration_buffer['current_state']
        previous_state = self.exploration_buffer['previous_state']
        action_probs = np.ones(self.action_dim)
        action_probs[-1] = 0.5

        # New state has not been seen, so downweight actions that you've
        # already taken in the current state
        if len(self.exploration_buffer['actions_made_in_curr_state']) > 0:
            for action in self.exploration_buffer['actions_made_in_curr_state']:
                action_probs[action] = 0
        else:
            pass

        action_probs = action_probs / action_probs.sum()
        action = np.random.choice(np.arange(self.action_dim), p=action_probs)
        zeros = np.zeros(self.action_dim)
        zeros[action] = 1
        return zeros

    def update_exploration_buffer(self, action, new_state):
        """Updates the exploration buffer with the given action and next state"""

        if not isinstance(new_state, list):
            new_state = new_state.tolist()
        new_state = tuple(new_state)
        if action is None:
            self.exploration_buffer['current_state'] = new_state
            self.exploration_buffer['last_action'] = None
            self.exploration_buffer['previous_state'] = None
            self.exploration_buffer['actions_made_in_curr_state'] = set()
        else:
            action = np.argmax(action)
            if self.exploration_buffer['current_state'] == new_state:
                self.exploration_buffer['actions_made_in_curr_state'].add(action)
                self.exploration_buffer['last_action'] = action
            else:
                self.exploration_buffer['previous_state'] = self.exploration_buffer['current_state']
                self.exploration_buffer['current_state'] = new_state
                self.exploration_buffer['last_action'] = action
                self.exploration_buffer['actions_made_in_curr_state'] = set()

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_dim)
        if not isinstance(state, list):
            state = state.tolist()
        current_state_pos = self.node_map_encoding_to_pos[tuple(state)]
        current_x, current_y = current_state_pos
        if action == 0:
            new_x, new_y = current_x - 1, current_y
        elif action == 1:
            new_x, new_y = current_x, current_y + 1
        elif action == 2:
            new_x, new_y = current_x + 1, current_y
        elif action == 3:
            new_x, new_y = current_x, current_y - 1
        else:
            new_x, new_y = current_x, current_y
        if new_x < 0 or new_x >= self.maze_dim or \
            new_y < 0 or new_y >= self.maze_dim:  # Out of bounds
            new_x, new_y = current_x, current_y
        if self.maze[new_x, new_y] == 0:  # Blocked state
            new_x, new_y = current_x, current_y
        new_state_pos = (new_x, new_y)
        new_state_encoding = self.node_map_pos_to_encoding[new_state_pos]
        if np.all(new_state_encoding == self.goal):
            reward = 1
        else:
            reward = 0
        return list(new_state_encoding), reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def to_networkx(self):
        """Converts the gridworld structure into a networkx Graph object.

        Returns:
            G (networkx.Graph): Graph representation of the gridworld where:
                - nodes have attributes: 'x', 'y', 'encoding'
                - edges represent valid movements between adjacent cells
        """
        import networkx as nx
        G = nx.Graph()

        # Add all valid nodes first
        for encoding, pos in self.node_map_encoding_to_pos.items():
            x, y = pos
            G.add_node(
                (x, y),  # use position tuple as node identifier
                x=x,
                y=y,
                encoding=encoding,
            )

        # Add edges between adjacent valid cells
        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                if self.maze[x, y] == 0:  # Skip blocked cells
                    continue

                # Check all adjacent cells (up, right, down, left)
                for dx, dy in [(-1,0), (0,1), (1,0), (0,-1)]:
                    new_x, new_y = x + dx, y + dy

                    # Skip if out of bounds
                    if new_x < 0 or new_x >= self.maze_dim or \
                       new_y < 0 or new_y >= self.maze_dim:
                        continue

                    # Add edge if adjacent cell is valid
                    if self.maze[new_x, new_y] == 1:
                        G.add_edge((x,y), (new_x,new_y))

        return G

class DarkroomEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.horizon = envs[0].horizon

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

    def deploy(self, ctrl, update_batch_online=False, return_max_rewards=False):
        if update_batch_online:
            raise NotImplementedError("Update batch online is not implemented for DarkroomEnvVec")

        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False
        device = ctrl.batch['context_states'].device
        full_trajectory = None

        if return_max_rewards:
            max_rewards = []
            for env in self._envs:
                dist_from_goal = env.dist_from_goal[tuple(env.state.tolist())]
                max_rewards.append(self.horizon - dist_from_goal + 1)

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

        if return_max_rewards:
            return obs, acts, next_obs, rews, max_rewards
        else:
            return obs, acts, next_obs, rews

if __name__ == '__main__':
    def make_new_darkroom():
        return DarkroomEnv(
            maze_dim=5,
            horizon=400,
            state_dim=10,
            node_encoding_corr=0.)
    
    env = make_new_darkroom()
    opt_action_map, dist_from_goal = env.make_opt_action_dict()
    import pdb; pdb.set_trace()