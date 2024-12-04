import os
import pickle
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.envs.trees import TreeEnv
from src.utils import (
    build_env_name,
    build_dataset_name,
)

n_search_seeds = 10000

@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_name = build_env_name(env_config)
    dataset_storage_dir = f'{cfg.storage_dir}/{cfg.wandb.project}/{env_name}/datasets'
    os.makedirs(dataset_storage_dir, exist_ok=True)

    unique_seeds = []
    unique_seeds_info = []
    for seed in range(n_search_seeds):
        try:
            env = TreeEnv(
                max_layers=env_config['max_layers'],
                initialization_seed=seed,
                horizon=env_config['horizon'],
                branching_prob=env_config['branching_prob'])
        except ValueError as e:
            if str(e) == "No leaves found in tree":
                continue
            else:
                raise e
        curr_env_goal = tuple(env.goal)
        curr_env_struct = sorted(env.node_map.keys())
        curr_seed_info = curr_env_struct + [curr_env_goal]

        new_info = True
        for prev_seed_info in unique_seeds_info:
            if prev_seed_info == curr_seed_info:
                new_info = False
                break
        if new_info:
            unique_seeds.append(seed)
            unique_seeds_info.append(curr_seed_info)

    print(f"Found {len(unique_seeds)} unique seeds")
    print("Truncating to 5000 seeds")
    unique_seeds_path = dataset_storage_dir + '/unique_seeds.pkl'
    with open(unique_seeds_path, 'wb') as f:
        pickle.dump(unique_seeds[:5000], f)

if __name__ == "__main__":
    main()
