#!/bin/bash

python eval_dqn_offline.py --config-name eval_q_table model.action_temp=0.05
python eval_dqn_offline.py --config-name eval_q_table model.action_temp=0.5
python eval_dqn_offline.py --config-name eval_q_table model.action_temp=1.0

