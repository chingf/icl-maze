import json
import os

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
wandb.login()

import numpy as np
import random
from src.dataset import Dataset, ImageDataset
from src.utils import (
    build_env_name,
    build_model_name,
    build_dataset_name,
)

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()

@hydra.main(version_base=None, config_path="configs", config_name="training")
def main(cfg: DictConfig):
    np.random.seed(0)  # TODO: figure out what's happening with seeds
    random.seed(0)
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['test'] = False
    model_config['horizon'] = env_config['horizon']
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config
    model = instantiate(model_config)
    model = model.to(device)


    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    train_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(0))
    test_dset_path = os.path.join(dataset_storage_dir, build_dataset_name(1))
    os.makedirs(model_storage_dir, exist_ok=True)

    # Random seed handling 
    tmp_seed = seed = cfg.seed  # TODO: figure out what's happening with seeds
    if seed == -1:
        tmp_seed = 0
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    # Set up datasets and dataloaders
    train_dataset = Dataset(train_dset_path, env_config)
    test_dataset = Dataset(test_dset_path, env_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, optimizer_config['batch_size'], shuffle=True,
        #num_workers=optimizer_config['num_workers'],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, optimizer_config['batch_size'],
        #num_workers=optimizer_config['num_workers'],
    )

    # Set up logging and checkpointing
    wandb_config = {
        'env': env_config,
        'model': model_config,
        'optimizer': optimizer_config,
        'seed': seed
    }
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=env_name + '/' + model_name,
        config=wandb_config,
        save_dir=cfg.storage_dir
    )

    # Save run ID for later evaluation
    run_id = wandb_logger.experiment.id
    if isinstance(run_id, str):
        run_info = {
            "run_id": str(wandb_logger.experiment.id),
            "run_name": str(wandb_logger.experiment.name),
        }
        with open(f'{model_storage_dir}/run_info.json', 'w') as f:
            json.dump(run_info, f)

    # Checkpoint top K models and last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_storage_dir,
        filename="{epoch}-{val_loss:.6f}",
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=optimizer_config['early_stopping_patience'],
        min_delta=optimizer_config['early_stopping_min_delta'],
        mode='min',
        verbose=True,
    )

    # Debugging callback
    class GradientDebugCallback(pl.Callback):
        def __init__(self):
            self.gradients_seen = set()
            
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self.gradients_seen.add(name)
                
        def on_train_epoch_end(self, trainer, pl_module):
            print("\nParameters that received gradients this epoch:")
            all_params = set(name for name, _ in pl_module.named_parameters())
            for name in all_params:
                if name in self.gradients_seen:
                    print(f"✓ {name}")
                else:
                    print(f"✗ {name}")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=optimizer_config['num_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        log_every_n_steps=None,
        precision="16-mixed",
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )


if __name__ == "__main__":
    main()