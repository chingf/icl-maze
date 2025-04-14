#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 24               # Number of cores (-c)
#SBATCH --mem=375G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate jax

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

python eval_offline.py
#python eval_offline_by_query_type.py
#python eval_qlearning_offline_by_query_type.py --config-name=eval_q_table
#python eval_qlearning_offline_by_query_type.py 
#python eval_qlearning_offline_by_query_type.py --config-name=eval_q_table model.action_temp=0.05
#python eval_qlearning_offline_by_query_type.py --config-name=eval_q_table model.action_temp=0.1
#python eval_qlearning_offline_by_query_type.py --config-name=eval_dqn model.action_temp=0.05
#python eval_qlearning_offline_by_query_type.py --config-name=eval_dqn model.action_temp=0.01
