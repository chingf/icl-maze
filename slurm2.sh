#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=32G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:4
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python train.py -m env.horizon=400 model.train_on_last_pred_only=False
python eval.py -m env.horizon=400 model.train_on_last_pred_only=False

#python train.py -m model.n_layer=8 model.n_embd=256 optimizer.batch_size=1024 env.horizon=400 env.layers=4
#python eval.py -m model.n_layer=8 model.n_embd=256 optimizer.batch_size=1024 env.horizon=400 env.layers=4
