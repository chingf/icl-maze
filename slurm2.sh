#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 72               # Number of cores (-c)
#SBATCH --mem=750G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:4
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python train.py env.n_envs=600000 env.horizon=1600 env.branching_prob=1.0 optimizer.batch_size=128
