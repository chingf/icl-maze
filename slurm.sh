#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 8               # Number of cores (-c)
#SBATCH --mem=20G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1
#SBATCH --mail-user=ching_fang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'-m model.n_layer=4
-m model.n_layer=3
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"


# run code
conda activate jax
python train.py $param_name
python eval.py $param_name
