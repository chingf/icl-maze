from dataclasses import dataclass
import os
import numpy as np

from src.utils import find_ckpt_file

fig_width = 6.4
fig_height = 4.8

import seaborn as sns
sns.set(
        font_scale=10/12.,
        palette='colorblind',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'black',  # Changed from lightgrey to black
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'black',  # Changed from dimgrey to black
            'axes.spines.right': True,   # Changed to True to show all spines
            'axes.spines.top': True,     # Changed to True to show all spines
            'text.color': 'black',       # Changed from dimgrey to black

            'lines.solid_capstyle': 'round',
            'lines.linewidth': 1,
            'legend.facecolor': 'white',
            'legend.framealpha': 0.8,

            'xtick.bottom': True,
            'xtick.color': 'black',      # Changed from dimgrey to black
            'xtick.direction': 'out',

            'ytick.color': 'black',      # Changed from dimgrey to black
            'ytick.direction': 'out',
            'ytick.left': True,

            'xtick.major.size': 2,
            'xtick.major.width': .5,
            'xtick.minor.size': 1,
            'xtick.minor.width': .5,

            'ytick.major.size': 2,
            'ytick.major.width': .5,
            'ytick.minor.size': 1,
            'ytick.minor.width': .5})

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']

def get_model_paths(corr, wandb_project="tree_maze", extra_flags=None):
    engram_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/icl-maze/"
    env_name = f"cntree_layers7_bprob0.9_corr{corr}_state_dim10_envs300000_H800_explore"
    if wandb_project == "tree_maze":
        if corr == 0.25:
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0.2_initseed1_batch512"
            #model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0_initseed4_batch512"
        elif corr == 0.0:
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0_initseed0_batch512"
        else:
            raise ValueError(f"Unknown correlation value: {corr}")
    elif wandb_project == "tree_maze_bigger_models":
        if corr == 0.25:
            if extra_flags is None:
                model_name = "transformer_end_query_embd1024_layer3_head4_lr0.0001_drop0_initseed3_batch256"
            elif extra_flags == 'layer4':
                model_name = "transformer_end_query_embd1024_layer4_head4_lr0.0001_drop0_initseed3_batch128"
            elif extra_flags == "layer6":
                model_name = f"transformer_end_query_embd512_layer6_head4_lr0.0001_drop0_initseed2_batch256"
            else:
                raise ValueError(f"Unknown extra flags: {extra_flags}")
        else:
            raise ValueError(f"Unknown correlation value: {corr}")
    elif wandb_project == "tree_maze_big_pretraining":
        env_name = f"cntree_layers7_bprob0.9_corr{corr}_state_dim10_envs600000_H800_explore"
        if corr == 0.25:
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0.2_initseed1_batch512"
        elif corr == 0.0:
            #model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0_initseed1_batch512"
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0.2_initseed2_batch512"
        else:
            raise ValueError(f"Unknown correlation value: {corr}")
    else:
        raise ValueError(f"Unknown wandb project: {wandb_project}")
    model_path = os.path.join(engram_dir, wandb_project, env_name, "models", model_name)
    ckpt_name = find_ckpt_file(model_path, "best")
    print(ckpt_name)
    path_to_pkl = os.path.join(model_path, ckpt_name)

    eval_dset_path = f"/n/holylfs06/LABS/krajan_lab/Lab/cfang/icl-maze/cntree/cntree_layers7_bprob1.0_corr{corr}_state_dim10_envs1000_H1600_explore/datasets/eval.pkl"
    return model_name, path_to_pkl, eval_dset_path