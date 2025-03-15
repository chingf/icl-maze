import numpy as np
import torch
from src.envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node:
    def __init__(
            self, layer, pos,
            parent=None, left=None, right=None,
            encoding_vector=None):
        self.layer = layer
        self.pos = pos
        self.left = left
        self.right = right
        self.parent = parent
        self.encoding_vector = encoding_vector

    def add_child(self, child, is_left=True):
        if is_left:
            self.left = child
        else:
            self.right = child
        child.parent = self

    def encoding(self):
        if self.encoding_vector is None:
            return (self.layer, self.pos)
        else:
            return self.encoding_vector

    def __str__(self):
        return f"({self.layer},{self.pos})"

class TreeEnv(BaseEnv):
    def __init__(
        self, max_layers, branching_prob, horizon,
        initialization_seed=None, node_encoding=None, goal=None
        ):
        """
        Args:
            max_layers: Maximum depth of the tree
            branching_prob: Probability of creating a child node at each position
            goal: Target (layer, pos) tuple
            horizon: Maximum steps per episode
        """
        self.max_layers = max_layers
        self.branching_prob = branching_prob
        self.horizon = horizon
        self.initialization_seed = initialization_seed
        self.node_encoding = node_encoding
        if initialization_seed is not None:
            np.random.seed(initialization_seed)
        if node_encoding is None:
            self.state_dim = 2
            self.node_encoding_bank = None
        elif node_encoding == 'random':
            self.state_dim = 10
            max_samples_needed = 2 ** self.max_layers - 1
            self.node_encoding_bank = np.random.randint(
                2, size=(max_samples_needed*2, self.state_dim)).astype(np.float32)
            self.node_encoding_bank = np.unique(self.node_encoding_bank, axis=0)
            self.node_encoding_bank = self.node_encoding_bank[~np.all(self.node_encoding_bank == 0, axis=1)]
            np.random.shuffle(self.node_encoding_bank)
            self.node_encoding_bank_idx = 0
            if self.node_encoding_bank.shape[0] < max_samples_needed:
                raise ValueError("Failed to generate enough unique node encodings")
        elif node_encoding == 'onehot':
            self.state_dim = 2 ** self.max_layers - 1
            self.node_encoding_bank = np.eye(self.state_dim)
            self.node_encoding_bank_idx = 0
        else:
            raise ValueError(f"Invalid node encoding: {node_encoding}")
        self.action_dim = 4  # (back, left, right, stay)
        self.root = None
        self.node_map = {}
        self.leaves = []
        self._generate_random_tree()
        if len(self.leaves) == 0:
            raise ValueError("No leaves found in tree")
        if goal is None:
            self.goal = self.sample_goal()
        else:
            if isinstance(goal, np.ndarray) and goal.size == self.state_dim:
                self.goal = goal
            else:
                self.goal = np.array(self._find_encoding_by_position(*goal))
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
        """Creates a new TreeEnv instance with identical parameters."""
        env = TreeEnv(
            max_layers=self.max_layers,
            branching_prob=self.branching_prob,
            horizon=self.horizon,
            initialization_seed=self.initialization_seed,
            node_encoding=self.node_encoding,
            goal=self.goal
        )
        env.reset_state_bank = self.reset_state_bank
        return env

    def _find_encoding_by_position(self, layer, pos):
        for node in self.node_map.values():
            if node.layer == layer and node.pos == pos:
                return node.encoding()
        raise ValueError(f"No node found at layer {layer} and position {pos}")
    
    def _sample_node_encoding(self):
        if self.node_encoding_bank is None:
            return None
        else:
            encoding = self.node_encoding_bank[self.node_encoding_bank_idx]
            self.node_encoding_bank_idx += 1
            return tuple(encoding.tolist())

    def _generate_random_tree(self):
        """Generates a random tree structure"""
        self.root = root = Node(0, 0, encoding_vector=self._sample_node_encoding())
        nodes_to_process = [(root, 1)]
        
        while nodes_to_process:
            current_node, current_layer = nodes_to_process.pop(0)
            self.node_map[current_node.encoding()] = current_node
            if current_layer == self.max_layers:
                self.leaves.append(current_node)
                continue
            elif current_layer > self.max_layers:
                raise ValueError("Likely an error has occurred in tree generation.")

            # Randomly decide to create left/right children
            if np.random.random() < self.branching_prob:
                left_child = Node(
                    current_layer,
                    2 * current_node.pos,
                    parent=current_node,
                    encoding_vector=self._sample_node_encoding())
                current_node.add_child(left_child, is_left=True)
                nodes_to_process.append((left_child, current_layer + 1))
            
            if np.random.random() < self.branching_prob:
                right_child = Node(
                    current_layer,
                    2 * current_node.pos + 1,
                    parent=current_node,
                    encoding_vector=self._sample_node_encoding())
                current_node.add_child(right_child, is_left=False)
                nodes_to_process.append((right_child, current_layer + 1))
        return root
    
    def sample_goal(self):
        goal = np.random.choice(self.leaves).encoding()
        return np.array(goal)
    
    def sample_state(self, from_origin=False):
        if from_origin:
            return np.array(list(self.root.encoding_vector))
        elif self.reset_state_bank is not None:
            state = self.reset_state_bank[np.random.choice(len(self.reset_state_bank))]
            return np.array(state)
        else:
            valid_states = list(self.node_map.keys())
            state = valid_states[np.random.choice(len(valid_states))]
            return np.array(state)

    def sample_action(self):
        i = np.random.randint(0, self.action_dim)
        a = np.zeros(self.action_dim)
        a[i] = 1
        return a

    def reset(self, from_origin=False):
        self.optimal_action_map, self.dist_from_goal = self.make_opt_action_dict()
        self.current_step = 0
        if from_origin:
            return list(self.root.encoding_vector)
        else:
            attempts = 0
            while True:
                state = self.sample_state()
                if self.dist_from_goal[tuple(state.tolist())] >= (self.max_layers - 1):
                    s = state.tolist()
                    break
                attempts += 1
                if attempts > 200:
                    import pdb; pdb.set_trace()
                    raise ValueError("Failed to sample a valid state")
            self.state = state
        return self.state

    def make_opt_action_dict(self):
        """Creates a dictionary mapping (current_node, target_node) to optimal action.
        Uses BFS to find shortest paths to each target."""
        target_node_encoding = tuple(self.goal.tolist())
        return self._make_opt_action_dict(target_node_encoding)

    def _make_opt_action_dict(self, target_node_encoding):
        target_node = self.node_map[target_node_encoding]
        queue = [target_node]  # (node, action_to_reach_it)
        visited = set()
        visited.add(target_node.encoding())
        opt_action_map = {target_node.encoding(): 3}
        dist_from_goal = {target_node.encoding(): 0}
        
        while queue:
            current_node = queue.pop(0)
            _current = current_node.encoding()
            current_dist = dist_from_goal[_current]

            # Check parent (action = 0)
            if current_node.parent:
                _parent = current_node.parent.encoding()
                if _parent not in visited:
                    visited.add(_parent)
                    if current_node.parent.left == current_node:
                        opt_action_map[_parent] = 1
                    else:
                        opt_action_map[_parent] = 2
                    dist_from_goal[_parent] = current_dist + 1
                    queue.append(current_node.parent)
            
            # Check left child (action = 1)
            if current_node.left:
                _left = current_node.left.encoding()
                if _left not in visited:
                    visited.add(_left)
                    opt_action_map[_left] = 0
                    dist_from_goal[_left] = current_dist + 1
                    queue.append(current_node.left)
            
            # Check right child (action = 2)
            if current_node.right:
                _right = current_node.right.encoding()
                if _right not in visited:
                    visited.add(_right)
                    opt_action_map[_right] = 0
                    dist_from_goal[_right] = current_dist + 1
                    queue.append(current_node.right)
        
        return opt_action_map, dist_from_goal

    def opt_action(self, state):
        """Returns optimal action to reach target (self.goal)"""
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
        else: # You just got to a new state
            # Try to go forward in an alternating manner
            if self.exploration_buffer['last_action'] == 1:
                action_probs[0] = 0.
                #action_probs[2] = 2
            elif self.exploration_buffer['last_action'] == 2:
                action_probs[0] = 0.
                #action_probs[1] = 2
            elif self.exploration_buffer['last_action'] == 0:
                current_state_node = self.node_map[current_state]
                previous_state_node = self.node_map[previous_state]
                if current_state_node.left == previous_state_node:
                    action_probs[1] = 0.
                    #action_probs[2] = 2
                elif current_state_node.right == previous_state_node:
                    action_probs[2] = 0.
                    #action_probs[1] = 2

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
        current_node = self.node_map[tuple(state)]
        if action == 0:  # Back
            if current_node.parent:
                state = current_node.parent.encoding()
        elif action == 1:  # Left
            if current_node.left:
                state = current_node.left.encoding()
        elif action == 2:  # Right
            if current_node.right:
                state = current_node.right.encoding()
        elif action == 3:  # Stay
            pass
        reward = 1 if np.all(state == self.goal) else 0
        return list(state), reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}
    
    def to_networkx(self, by_node_encoding=False):
        """Converts the tree structure into a networkx Graph object.
        
        Returns:
            G (networkx.Graph): Graph representation of the tree where:
                - nodes have attributes: 'layer', 'pos', 'encoding'
                - edges represent parent-child relationships
        """
        import networkx as nx
        G = nx.Graph()

        def get_identifier(node):
            if by_node_encoding:
                return node.encoding()
            else:
                return (node.layer, node.pos)
        
        # Add all nodes first
        for encoding, node in self.node_map.items():
            G.add_node(
                get_identifier(node),  # use encoding as node identifier
                layer=node.layer,
                pos=node.pos,
                encoding=encoding,
            )
        
        # Add edges by traversing from root
        def add_edges(node):
            if node.left:
                G.add_edge(get_identifier(node), get_identifier(node.left))
                add_edges(node.left)
            if node.right:
                G.add_edge(get_identifier(node), get_identifier(node.right))
                add_edges(node.right)
                
        add_edges(self.root)
        return G


class TreeEnvVec(BaseEnv):
    """
    Vectorized Tree environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.horizon = envs[0].horizon
        self.online_batch_size = None

    def reset(self):
        print(f"Resetting all environments")
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

    def deploy(
            self, ctrl, update_batch_online=False,
            return_max_rewards=False
            ):

        if update_batch_online and self.online_batch_size is None:
            raise ValueError("online_batch_size is not set")
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
                max_rewards.append(self.horizon - dist_from_goal)

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

            if update_batch_online:
                if full_trajectory is None:
                    full_trajectory = ctrl.batch
                new_context_states = torch.tensor(np.array(obs[-1]), device=device, dtype=torch.float32)
                new_context_actions = torch.tensor(np.array(acts[-1]), device=device, dtype=torch.float32)
                new_context_next_states = torch.tensor(np.array(next_obs[-1]), device=device, dtype=torch.float32)
                new_context_rewards = torch.tensor(np.array(rews[-1]).reshape((-1,1)), device=device, dtype=torch.float32)
                full_trajectory['context_states'] = torch.cat((
                    full_trajectory['context_states'], new_context_states[:, None, :]), dim=1)
                full_trajectory['context_actions'] = torch.cat((
                    full_trajectory['context_actions'], new_context_actions[:, None, :]), dim=1)
                full_trajectory['context_next_states'] = torch.cat((
                    full_trajectory['context_next_states'], new_context_next_states[:, None, :]), dim=1)
                full_trajectory['context_rewards'] = torch.cat((
                    full_trajectory['context_rewards'], new_context_rewards[:, None, :]), dim=1)
                batch = {}
                for key in [
                        'context_states', 'context_actions',
                        'context_next_states', 'context_rewards']:
                    batch[key] = full_trajectory[key][:, -self.online_batch_size:, :]
                ctrl.set_batch(batch)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)

        if return_max_rewards:
            return obs, acts, next_obs, rews, max_rewards
        else:
            return obs, acts, next_obs, rews
        
if __name__ == '__main__':
    def make_new_tree():
        return TreeEnv(
            max_layers=7,
            branching_prob=1.0,
            horizon=800,
            node_encoding='random',
        )

    tree = make_new_tree()

    import pickle
    import os
    abs_path = '/n/home04/cfang/Code/icl-maze/src/envs/'
    dist_mat = pickle.load(open(
        os.path.join(abs_path, f'depth{tree.max_layers}_distance_matrix.pkl'), 'rb'
        ))
    ordered_encodings = [np.nan*np.ones(tree.state_dim) for _ in range(dist_mat.shape[0])]
    idxs = []
    for encoding in tree.node_map:
        node = tree.node_map[encoding]
        layer = node.layer
        pos = node.pos
        if layer == pos == 0:
            idx = 0
        else:
            idx = (2**layer)-1 + pos
        ordered_encodings[idx] = np.array(encoding)
        idxs.append(idx)
    print(idxs)
    ordered_encodings = np.array(ordered_encodings)
    CC = np.corrcoef(ordered_encodings)

    import matplotlib.pyplot as plt
    plt.imshow(CC, cmap='jet')
    plt.colorbar()
    plt.savefig('cc.png')
    plt.show()

    plt.imshow(-dist_mat, cmap='jet')
    plt.colorbar()
    plt.savefig('dist_mat.png')
    plt.show()
