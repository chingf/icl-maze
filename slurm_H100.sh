#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 48               # Number of cores (-c)
#SBATCH --mem=750G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:2
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

#python train.py --config-name=training_darkroom
python train.py --config-name=training_darkroom model.initialization_seed=0
#python train.py env.node_encoding_corr=0.25 model.n_layer=6 model.dropout=0.2 optimizer.batch_size=256

#python train.py wandb.project=from_middle override_dataset_dir=null 
#python train.py env.node_encoding_corr=0.25
#python train.py model.dropout=0.2 optimizer.lr=1e-4 optimizer.use_scheduler=True model.initialization_seed=1
#python train.py model.dropout=0.2 optimizer.lr=1e-4 optimizer.use_scheduler=True model.initialization_seed=2
