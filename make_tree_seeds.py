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

    unique_seeds_info = {}
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
        env_goal = tuple(env.goal.tolist())
        env_struct = tuple(sorted(env.node_map.keys()))

        if env_struct not in unique_seeds_info.keys():
            unique_seeds_info[env_struct] = {}
        unique_seeds_info[env_struct][env_goal] = seed

    # Shuffle the environments found
    unique_structs = list(unique_seeds_info.keys())
    random.shuffle(unique_structs)

    # Determine how many seeds to allocate to train, test, and eval
    goal_num_train_seeds = int(5000*0.8)
    goal_num_test_seeds = int(5000*0.1)
    goal_num_eval_seeds = int(5000*0.1)

    # Decide how to split up the environments into train/test/eval
    split_idxs = []
    n_envs_seen = 0
    goal_num_seeds = [goal_num_train_seeds, goal_num_test_seeds, goal_num_eval_seeds]
    for struct_idx, unique_struct in enumerate(unique_structs):
        n_goals = len(unique_seeds_info[unique_struct].keys())
        n_envs_seen += n_goals
        if n_envs_seen >= goal_num_seeds[0]:
            split_idxs.append(struct_idx+1)
            n_envs_seen = 0
            goal_num_seeds.pop(0)
            if len(goal_num_seeds) == 0:
                break

    # Now that keys are split into train/test/eval, unpack and flattent the seeds
    train_seed_keys = unique_structs[:split_idxs[0]]
    test_seed_keys = unique_structs[split_idxs[0]:split_idxs[1]]
    eval_seed_keys = unique_structs[split_idxs[1]:split_idxs[2]]
    train_seeds = all_items(unique_seeds_info, train_seed_keys)
    test_seeds = all_items(unique_seeds_info, test_seed_keys)
    eval_seeds = all_items(unique_seeds_info, eval_seed_keys) 

    total_seeds = len(train_seeds) + len(test_seeds) + len(eval_seeds)
    print(f"Found {total_seeds} unique seeds")
    unique_seeds_path = dataset_storage_dir + '/unique_seeds.pkl'
    with open(unique_seeds_path, 'wb') as f:
        pickle.dump({
            'train': train_seeds,
            'test': test_seeds,
            'eval': eval_seeds
        }, f)

def all_items(unique_seeds_info, chosen_keys):
    seeds = []
    for key in chosen_keys:
        struct_dict = unique_seeds_info[key]
        for goal, seed in struct_dict.items():
            seeds.append(seed)
    return seeds

if __name__ == "__main__":
    main()
