import pickle
import h5py
import numpy as np
import torch
import sys
import signal

from src.utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(path, config):
    if path.lower().endswith(('.h5', '.hdf5', '.he5')):
        return HDF5Dataset(path, config)
    else:
        return Dataset(path, config)
    
class HDF5Dataset(torch.utils.data.Dataset):
    _open_files = set()

    def __init__(self, path, config):
        self.h5_path = path
        self._file = None
        self.horizon = config['horizon']
        with h5py.File(path, 'r') as f:
            self.length = len(f.keys())
            self.indices = list(f.keys())
        self.epoch = 0
        self.zeros = np.zeros(config['state_dim'] ** 2 + config['action_dim'] + 1)
        self.zeros = convert_to_tensor(self.zeros)

        self._setup_signal_handlers()
    
    @classmethod
    def _setup_signal_handlers(cls):
        # Only set up handlers once
        if not hasattr(cls, '_handlers_setup'):
            signal.signal(signal.SIGINT, cls._signal_handler)
            signal.signal(signal.SIGTERM, cls._signal_handler)
            cls._handlers_setup = True

    @classmethod
    def _signal_handler(cls, signum, frame):
        """Handle cleanup when receiving a signal"""
        print("\nClosing HDF5 files...")
        for file in cls._open_files:
            try:
                file.close()
            except Exception as e:
                print(f"Error closing file: {e}")
        cls._open_files.clear()
        sys.exit(1) # Re-raise the signal

    def shuffle_indices(self):
        np.random.shuffle(self.indices)
        self.epoch += 1
    
    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
            self._open_files.add(self._file)

        shuffled_idx = self.indices[idx]
        entry = self._file[shuffled_idx]
        res = {
            'context_states': convert_to_tensor(entry['context_states']),
            'context_actions': convert_to_tensor(entry['context_actions']),
            'context_next_states': convert_to_tensor(entry['context_next_states']),
            'context_rewards': convert_to_tensor(entry['context_rewards'])[:, None],
            'query_states': convert_to_tensor(entry['query_state']),
            'optimal_actions': convert_to_tensor(entry['optimal_action']),
            'zeros': self.zeros,
        }

        return res

    def __len__(self):
        return self.length

    def __del__(self):
        if self._file is not None:
            self._file.close()
            self._open_files.discard(self._file)

        
class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])
            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        return res
