import numpy as np
import torch
import os
import pickle
from src.envs.base_env import BaseEnv
from src.envs.trees import TreeEnv, Node

abs_path = '/n/home04/cfang/Code/icl-maze/src/envs/'  # so hacky

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CnTreeEnv(TreeEnv):
    def __init__(
        self, max_layers, branching_prob, horizon,
        initialization_seed=None, goal=None, node_encoding_corr=0, state_dim=256
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
        self.node_encoding_corr = node_encoding_corr
        self.action_dim = 4  # (back, left, right, stay)
        self.state_dim = state_dim

        # Random generation of tree and encodings
        if initialization_seed is not None:
            np.random.seed(initialization_seed)
        self.root = None
        self.node_map = {}
        self.leaves = []
        self._generate_random_tree()
        if goal is None:
            self.goal = self.sample_goal()
        else:
            if isinstance(goal, np.ndarray) and goal.size == self.state_dim:
                self.goal = goal
            else:
                self.goal = np.array(self._find_encoding_by_position(*goal))
        np.random.seed()

        # Variables used for exploration or pretraining dataset generation
        self.optimal_action_map = None
        self.dist_from_goal = None
        self.exploration_buffer = {  # Only used when generating exploratory traj.
            'actions_made_in_curr_state': set(),
            'previous_state': None,
            'current_state': None,
            'last_action': None,
            }
        self.forbidden_states = None  # TODO: remove later, useful for debugging
        
    def clone(self):
        """Creates a new TreeEnv instance with identical parameters."""
        return TreeEnv(
            max_layers=self.max_layers,
            branching_prob=self.branching_prob,
            horizon=self.horizon,
            initialization_seed=self.initialization_seed,
            node_encoding_corr=self.node_encoding_corr,
            state_dim=self.state_dim,
            goal=self.goal
        )
    
    def _sample_node_encoding(self, layer, pos, expansion_mat, dist_mat):
        if layer == pos == 0:
            curr_loc = 0
        else:
            curr_loc = (2**layer)-1 + pos
        encoding_vector = expansion_mat @ dist_mat[curr_loc]
        encoding_vector = encoding_vector / np.linalg.norm(encoding_vector)
        return tuple(encoding_vector.tolist())

    def _generate_random_tree(self):
        """Generates a random tree structure"""
        expansion_mat = np.random.randn(self.state_dim, 2**self.max_layers-1)
        #expansion_mat[np.abs(expansion_mat) < 1.5] = 0

        #expansion_mat = np.random.choice([-1, 1], size=(self.state_dim, 2**self.max_layers-1))

        dist_mat = pickle.load(open(
            os.path.join(abs_path, f'depth{self.max_layers}_distance_matrix.pkl'), 'rb'
            ))
        # Apply correlation based on geodesic distance
        dist_mat = np.power(self.node_encoding_corr, dist_mat)
        self.root = root = Node(
            0, 0, encoding_vector=self._sample_node_encoding(0, 0, expansion_mat, dist_mat))
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
                child_encoding_vector = self._sample_node_encoding(
                    current_layer, 2*current_node.pos, expansion_mat, dist_mat)
                left_child = Node(
                    current_layer,
                    2 * current_node.pos,
                    parent=current_node,
                    encoding_vector=child_encoding_vector)
                current_node.add_child(left_child, is_left=True)
                nodes_to_process.append((left_child, current_layer + 1))
            
            if np.random.random() < self.branching_prob:
                child_encoding_vector = self._sample_node_encoding(
                    current_layer, 2*current_node.pos+1, expansion_mat, dist_mat)
                right_child = Node(
                    current_layer,
                    2 * current_node.pos + 1,
                    parent=current_node,
                    encoding_vector=child_encoding_vector)
                current_node.add_child(right_child, is_left=False)
                nodes_to_process.append((right_child, current_layer + 1))

        if len(self.leaves) == 0:
            raise ValueError("No leaves found in tree")
        return root
    
        
if __name__ == '__main__':
    def make_new_tree():
        return CnTreeEnv(
            max_layers=7,
            branching_prob=1.0,
            horizon=800,
            node_encoding_corr=0.6,
            state_dim=10
        )

    tree = make_new_tree()

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
    plt.figure()
    plt.imshow(CC, cmap='jet')
    plt.colorbar()
    plt.savefig('cc.png')
    plt.show()

    plt.figure()
    plt.imshow(-dist_mat, cmap='jet')
    plt.colorbar()
    plt.savefig('dist_mat.png')
    plt.show()
