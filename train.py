import json
import os
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule
import wandb
from src.dataset import get_dataset, HDF5Dataset
from src.utils import (
    build_env_name,
    build_model_name,
    ShuffleIndicesCallback
)

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
wandb.login()

@hydra.main(version_base=None, config_path="configs", config_name="training")
def main(cfg: DictConfig):
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['test'] = False
    model_config['state_dim'] = env_config['state_dim']
    model_config['action_dim'] = env_config['action_dim']
    model_config['optimizer_config'] = optimizer_config
    model = instantiate(model_config)
    model = model.to(device)

    # Directory path handling
    env_name = build_env_name(env_config)
    model_name = build_model_name(model_config, optimizer_config)
    if cfg.override_dataset_dir is not None:
        dataset_storage_dir = f'{cfg.storage_dir}/{cfg.override_dataset_dir}/{env_name}/datasets'
    else:
        dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    model_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/models/{model_name}'
    # Check if train dataset is pickle or h5 format
    use_h5 = os.path.exists(os.path.join(dataset_storage_dir, 'train.h5'))
    file_suffix = '.h5' if use_h5 else '.pkl'
    train_dset_path = os.path.join(dataset_storage_dir, 'train' + file_suffix)
    test_dset_path = os.path.join(dataset_storage_dir, 'test' + file_suffix)
    os.makedirs(model_storage_dir, exist_ok=True)

    # Set up logging and checkpointing
    wandb_config = {
        'env': env_config,
        'model': model_config,
        'optimizer': optimizer_config,
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
    with open(f'{model_storage_dir}/model_config.json', 'w') as f:
        json.dump(model_config, f)

    # Checkpoint top K models and last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_storage_dir,
        filename="{epoch}-{val_loss:.6f}",
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    # Dataset shuffler callback
    shuffle_indices_callback = ShuffleIndicesCallback()

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=optimizer_config['num_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, shuffle_indices_callback],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        log_every_n_steps=None,
        precision="16-mixed",
    )

    try:
        # Set up datasets and dataloaders
        datamodule = DataModule(
            train_dset_path=train_dset_path,
            test_dset_path=test_dset_path,
            env_config=env_config,
            batch_size=optimizer_config['batch_size'],
            optimizer_config=optimizer_config
        )
    
        # Train model
        trainer.fit(
            model,
            datamodule=datamodule,
        )
    finally:
        for file in HDF5Dataset._open_files.copy():
            try:
                file.close()
            except Exception as e:
                print(f"Error closing file: {e}")
        HDF5Dataset._open_files.clear()

class DataModule(LightningDataModule):
    def __init__(
            self,
            train_dset_path,
            test_dset_path,
            env_config,
            batch_size,
            optimizer_config
    ):
        super().__init__()
        self.train_dset_path = train_dset_path
        self.test_dset_path = test_dset_path
        self.env_config = env_config
        self.batch_size = batch_size
        self.optimizer_config = optimizer_config

    def setup(self, stage=None):
        self.train_dataset = get_dataset(self.train_dset_path, self.env_config)
        self.test_dataset = get_dataset(self.test_dset_path, self.env_config)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, self.batch_size,
            shuffle=(not isinstance(self.train_dataset, HDF5Dataset)),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, self.batch_size,
        )
    
    def teardown(self, stage=None):
        """Clean up when training ends"""
        if self.train_dataset:
            if hasattr(self.train_dataset, '_file') and self.train_dataset._file is not None:
                self.train_dataset._file.close()
                HDF5Dataset._open_files.discard(self.train_dataset._file)
        if self.test_dataset:
            if hasattr(self.test_dataset, '_file') and self.test_dataset._file is not None:
                self.test_dataset._file.close()
                HDF5Dataset._open_files.discard(self.test_dataset._file)


if __name__ == "__main__":
    main()