#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 48               # Number of cores (-c)
#SBATCH --mem=375G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:2
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python eval.py
#python train.py model.dropout=0 model.n_layer=2 model.n_embd=1024 optimizer.lr=0.001
#python eval.py model.dropout=0 model.n_layer=2 #model.n_embd=1024 optimizer.lr=0.001
#python eval.py model.dropout=0 model.n_layer=3
