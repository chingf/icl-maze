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
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']

def get_model_paths(corr, wandb_project="darkroom_simple", extra_flags=None):
    engram_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/icl-maze/"
    env_name = f"darkroom_dim5_corr{corr}_state_dim10_envs900000_H200_explore"
    if wandb_project == "darkroom_simple":
        if corr == 0.25:
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0.2_initseed2_batch1024"
            model_name = "transformer_end_query_embd512_layer3_head4_lr0.0001_drop0.0_initseed2_batch1024"
        else:
            raise ValueError(f"Unknown correlation value: {corr}")
    else:
        raise ValueError(f"Unknown wandb project: {wandb_project}")
    model_path = os.path.join(engram_dir, wandb_project, env_name, "models", model_name)
    ckpt_name = find_ckpt_file(model_path, "best")
    print(ckpt_name)
    path_to_pkl = os.path.join(model_path, ckpt_name)

    eval_dset_path = f"/n/holylfs06/LABS/krajan_lab/Lab/cfang/icl-maze/{wandb_project}/{env_name}/datasets/eval.pkl"
    return model_name, path_to_pkl, eval_dset_path