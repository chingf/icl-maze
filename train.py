import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import numpy as np
import random
from src.dataset import Dataset, ImageDataset
from src.utils import (
    build_data_filename,
    build_model_filename,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = instantiate(model_config)
    model = model.to(device)

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

    # Set up directories
    path_train = build_data_filename(env_config, mode=0)
    path_test = build_data_filename(env_config, mode=1)
    model_chkpt_path = build_model_filename(env_config, model_config)
    os.makedirs(f'pickles/models/{model_chkpt_path}', exist_ok=True)
    os.makedirs(f'figs/{model_chkpt_path}/loss', exist_ok=True)

    # Set up datasets and dataloaders
    train_dataset = Dataset(path_train, env_config)
    test_dataset = Dataset(path_test, env_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, optimizer_config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, optimizer_config['batch_size'], shuffle=True)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay']
        )
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # Training loop
    test_loss = []
    train_loss = []
    print("Num train batches: " + str(len(train_loader)))
    print("Num test batches: " + str(len(test_loader)))
    for epoch in range(optimizer_config['num_epochs']):
        # EVALUATION
        print(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions']
                pred_actions = model(batch)
                true_actions = true_actions.unsqueeze(
                    1).repeat(1, pred_actions.shape[1], 1)
                true_actions = true_actions.reshape(-1, env_config['action_dim'])
                pred_actions = pred_actions.reshape(-1, env_config['action_dim'])

                loss = loss_fn(pred_actions, true_actions)
                epoch_test_loss += loss.item() / env_config['horizon']

        test_loss.append(epoch_test_loss / len(test_dataset))
        end_time = time.time()
        print(f"\tTest loss: {test_loss[-1]}")
        print(f"\tEval time: {end_time - start_time}")

        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            pred_actions = model(batch)
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_actions.shape[1], 1)
            true_actions = true_actions.reshape(-1, env_config['action_dim'])
            pred_actions = pred_actions.reshape(-1, env_config['action_dim'])

            optimizer.zero_grad()
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / env_config['horizon']

        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        print(f"\tTrain loss: {train_loss[-1]}")
        print(f"\tTrain time: {end_time - start_time}")

        # LOGGING
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       f'pickles/models/{model_chkpt_path}/epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}")
            print(f"Test Loss:        {test_loss[-1]}")
            print(f"Train Loss:       {train_loss[-1]}")
            print("\n")

            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/{model_chkpt_path}/loss/train_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'pickles/models/{model_chkpt_path}/final.pt')
    print("Done.")


if __name__ == "__main__":
    main()