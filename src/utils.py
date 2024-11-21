import numpy as np
import torch
import os

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
    
def build_data_filename(env_config, mode, storage_dir=None):
    if env_config['env'] == 'darkroom':
        filename = build_darkroom_data_filename(env_config, mode)
        if storage_dir is not None:
            filename = os.path.join(storage_dir, filename)
        return filename
    else:
        raise NotImplementedError
    
def build_model_filename(env_config, model_config, optimizer_config):
    if env_config['env'] == 'darkroom':
        return build_darkroom_model_filename(env_config, model_config, optimizer_config)
    else:
        raise NotImplementedError

def build_darkroom_data_filename(env_config, mode):
    """
    Builds the filename for the darkroom data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename = env_config['env']
    filename += '_envs' + str(env_config['n_envs'])
    filename += '_H' + str(env_config['horizon'])
    filename += '_d' + str(env_config['dim'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_' + env_config['rollin_type']
        filename += '_eval'
        
    return filename + '.pkl'


def build_darkroom_model_filename(env_config, model_config, optimizer_config):
    """
    Builds the filename for the darkroom model.
    """

    env_filename = env_config['env']
    env_filename += '_envs' + str(env_config['n_envs'])
    env_filename += '_H' + str(env_config['horizon'])
    env_filename += '_d' + str(env_config['dim'])

    model_filename = model_config['name']
    model_filename += '_embd' + str(model_config['n_embd'])
    model_filename += '_layer' + str(model_config['n_layer'])
    model_filename += '_head' + str(model_config['n_head'])
    model_filename += '_do' + str(model_config['dropout'])
    model_filename += '_lr' + str(optimizer_config['lr'])
    model_filename += '_batch' + str(optimizer_config['batch_size'])

    return env_filename + '/' + model_filename
