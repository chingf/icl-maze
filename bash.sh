#!/bin/bash

python make_tree_seeds.py
python collect_data.py
python train.py

python make_tree_seeds.py env.horizon=600
python collect_data.py env.horizon=600
python train.py env.horizon=600

python train.py model.n_head=8

python eval.py
python eval.py env.horizon=600
python eval.py model.n_head=8
