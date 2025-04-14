#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=250G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python eval_offline_by_query_type.py wandb.project=tree_maze_big_pretraining env.n_envs=600000 model.dropout=0.2 model.initialization_seed=2
python eval_offline_by_query_type.py wandb.project=tree_maze_big_pretraining env.n_envs=600000 env.node_encoding_corr=0.25 model.dropout=0.2 model.initialization_seed=1
