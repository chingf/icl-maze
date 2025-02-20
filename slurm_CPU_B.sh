#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 32               # Number of cores (-c)
#SBATCH --mem=512G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:0
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python make_tree_seeds.py env.node_encoding_corr=0.
python collect_data.py env.node_encoding_corr=0.


