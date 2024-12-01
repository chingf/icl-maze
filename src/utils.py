import numpy as np
import torch
import os
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    elif env_filename == 'darkroom':
        env_filename += '_dim' + str(env_config['dim'])
        env_filename += '_envs' + str(env_config['n_envs'])
        env_filename += '_H' + str(env_config['horizon'])
        env_filename += '_' + env_config['rollin_type']
    return env_filename

def build_model_name(model_config, optimizer_config):
    model_filename = model_config['name']
    model_filename += '_embd' + str(model_config['n_embd'])
    model_filename += '_layer' + str(model_config['n_layer'])
    model_filename += '_head' + str(model_config['n_head'])
    model_filename += '_lr' + str(optimizer_config['lr'])
    model_filename += '_batch' + str(optimizer_config['batch_size'])
    return model_filename

def build_dataset_name(mode):
    if mode == 0:
        filename = 'train'
    elif mode == 1:
        filename = 'test'
    elif mode == 2:
        filename = 'eval'
    return filename + '.pkl'
