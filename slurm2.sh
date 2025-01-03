#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 32               # Number of cores (-c)
#SBATCH --mem=256G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:4
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python make_tree_seeds.py env.goal_total_seeds=100000 env.n_search_seeds=300000
python collect_data.py env.horizon=500
python train.py model.n_embd=1024 model.dropout=0.25 optimizer.batch_size=64 model.n_head=8 model.n_layer=8

