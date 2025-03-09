#!/bin/bash

python train.py model.initialization_seed=0 optimizer.use_scheduler=False optimizer.lr=1e-5 optimizer.num_epochs=100
python train.py model.initialization_seed=1 optimizer.use_scheduler=False optimizer.lr=1e-5 optimizer.num_epochs=100
python train.py model.initialization_seed=2 optimizer.use_scheduler=False optimizer.lr=1e-5 optimizer.num_epochs=100