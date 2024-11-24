#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 8               # Number of cores (-c)
#SBATCH --mem=20G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python train.py -m model.dropout=0.2
python eval.py -m model.dropout=0.2

python train.py -m model.n_head=1 model.dropout=0.2
python eval.py -m model.n_head=1 model.dropout=0.2

python train.py -m env.rollin_type=expert model.dropout=0.2
python eval.py -m env.rollin_type=expert model.dropout=0.2

python train.py -m env.rollin_type=expert model.n_head=1 model.dropout=0.2
python eval.py -m env.rollin_type=expert model.n_head=1 model.dropout=0.2
