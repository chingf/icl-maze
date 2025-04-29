import numpy as np
import torch
import os
import glob
import random
from pytorch_lightning.callbacks import Callback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Generally useful functions

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)

def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()
    
def set_all_seeds(seed=None):
    if seed == None:
        torch.seed()
    else:
        torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


## Things for training
class ShuffleIndicesCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer.datamodule.train_dataset, 'indices'):
            trainer.datamodule.train_dataset.shuffle_indices()


## Functions for evaluation
    
def find_ckpt_file(ckpt_dir, epoch):
    if isinstance(epoch, str):
        if epoch != "best":
            raise ValueError("Unrecognized epoch string name")
        epoch_pattern = f'{ckpt_dir}/epoch=*-val_loss=*.ckpt'
        matching_files = glob.glob(epoch_pattern)
        if len(matching_files) == 0:
            raise ValueError(f"No checkpoint found.")
        val_losses = []
        for f in matching_files:
            val_loss = float(f.split('val_loss=')[1].split('.ckpt')[0])
            val_losses.append(val_loss)
        best_idx = np.argmin(val_losses)
        ckpt_name = os.path.basename(matching_files[best_idx])
    elif epoch > 0:
        epoch_pattern = f'{ckpt_dir}/epoch={epoch}-val_loss=*.ckpt'
        matching_files = glob.glob(epoch_pattern)
        if len(matching_files) == 0:
            raise ValueError(f"No checkpoint found for epoch {epoch}")
        ckpt_name = os.path.basename(matching_files[0])
    else:
        # Check for versioned 'last' checkpoints (last-v1, last-v2, etc)
        last_pattern = f'{ckpt_dir}/last-v*.ckpt'
        versioned_files = glob.glob(last_pattern)
        if versioned_files:
            # Extract version numbers and find highest
            versions = []
            for f in versioned_files:
                v = int(f.split('last-v')[1].split('.ckpt')[0])
                versions.append(v)
            highest_version = max(versions)
            ckpt_name = f'last-v{highest_version}.ckpt'
        else:
            ckpt_name = 'last.ckpt'  # Use last checkpoint if epoch not specified
    return ckpt_name


## Filename generation

def build_env_name(env_config):
    env_filename = env_config['env']
    if env_filename == 'maze':
        env_filename += '_layers' + str(env_config['layers'])
        env_filename += '_envs' + str(env_config['n_envs'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    elif 'darkroom' in env_filename:
        env_filename += '_dim' + str(env_config['maze_dim'])
        env_filename += '_corr' + str(env_config['node_encoding_corr'])
        env_filename += '_state_dim' + str(env_config['state_dim'])
        env_filename += '_envs' + str(env_config['n_envs'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    elif env_filename == 'tree_lazy_loading':
        env_filename += '_layers' + str(env_config['max_layers'])
        env_filename += '_bprob' + str(env_config['branching_prob'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    elif env_filename == 'tree' or env_filename == 'tree_origin':
        env_filename += '_layers' + str(env_config['max_layers'])
        env_filename += '_bprob' + str(env_config['branching_prob'])
        env_filename += '_envs' + str(env_config['n_envs'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    elif env_filename == 'cntree':
        env_filename += '_layers' + str(env_config['max_layers'])
        env_filename += '_bprob' + str(env_config['branching_prob'])
        env_filename += '_corr' + str(env_config['node_encoding_corr'])
        env_filename += '_state_dim' + str(env_config['state_dim'])
        env_filename += '_envs' + str(env_config['n_envs'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    return env_filename

def build_model_name(model_config, optimizer_config):
    model_filename = model_config['name']
    if 'transformer' in model_filename:
        model_filename += '_embd' + str(model_config['n_embd'])
        model_filename += '_layer' + str(model_config['n_layer'])
        model_filename += '_head' + str(model_config['n_head'])
        model_filename += '_lr' + str(optimizer_config['lr'])
        model_filename += '_drop' + str(model_config['dropout'])
        model_filename += '_initseed' + str(model_config['initialization_seed'])
        model_filename += '_batch' + str(optimizer_config['batch_size'])
        if optimizer_config['use_scheduler'] == False:
            model_filename += '_nosched'
    elif 'rnn' in model_filename:
        model_filename += '_embd' + str(model_config['n_embd'])
        model_filename += '_layer' + str(model_config['n_layer'])
        model_filename += '_dropout' + str(model_config['dropout'])
        model_filename += '_lr' + str(optimizer_config['lr'])
        model_filename += '_batch' + str(optimizer_config['batch_size'])
    elif model_filename == 'dqn':
        model_filename += '_nlayers' + str(model_config['n_layers'])
        model_filename += '_gamma' + str(model_config['gamma'])
        model_filename += '_target' + str(model_config['target_update'])
        model_filename += '_lr' + str(optimizer_config['lr'])
    elif model_filename == 'q_table':
        model_filename += '_gamma' + str(model_config['gamma'])
    return model_filename

def build_dataset_name(mode):
    if mode == 0:
        filename = 'train'
    elif mode == 1:
        filename = 'test'
    elif mode == 2:
        filename = 'eval'
    return filename + '.pkl'


## Environment Visualizers
def print_tree(tree):
    """Prints a visual ASCII representation of the tree structure."""

    height = tree.max_layers
    width = (2 ** (height)) * 4
    
    # Process nodes level by level
    queue = [(tree.root, 0, width // 2)]  # (node, level, horizontal_position)
    current_level = 0
    level_nodes = []
    
    while queue:
        node, level, pos = queue.pop(0)
        
        if (level != current_level) and (len(level_nodes) > 0):
            line = " " * width # Print the current level's nodes
            for n, p in level_nodes:
                node_str = str(n)
                # Center the node string around position p
                start_pos = max(0, p - len(node_str) // 2)
                line = line[:start_pos] + node_str + line[start_pos + len(node_str):]
            print(line)

            line = " " * width # Print connecting lines
            for n, p in level_nodes:
                if n.left:
                    line = line[:p-2] + "/" + line[p-1:]
                if n.right:
                    line = line[:p+1] + "\\" + line[p+2:]
            print(line)
            
            level_nodes = []
            current_level = level
        
        level_nodes.append((node, pos))
        
        # Calculate positions for children
        spacing = width // (2 ** (level + 2))
        if node.left:
            queue.append((node.left, level + 1, pos - spacing))
        if node.right:
            queue.append((node.right, level + 1, pos + spacing))
    
    # Print the last level
    if len(level_nodes) > 0:
        line = " " * width
        for n, p in level_nodes:
            node_str = str(n)
            start_pos = max(0, p - len(node_str) // 2)
            line = line[:start_pos] + node_str + line[start_pos + len(node_str):]
        print(line)