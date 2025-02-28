# Overview
This is an adapted version of the [decision pretrained transformer](https://github.com/jon--lee/decision-pretrained-transformer/tree/main). You will need Hydra, Torch Lightning, and WandB.

## Important src files
- `src/envs/cntrees.py`: Tree-based mazes with Gaussian-sampled sensory inputs. Currently the main environment I'm using, the other stuff is older things I tried.
- `src/models/transformer_end_query.py`: The main model file. Wraps around `src.models.simple_gpt2` which is a modified version of the transformers library GPT2 model. This is because I needed to pass in custom attention masks during training to allow for variable context length training while still enforcing that the query token comes last. There's other scattered code for other transformer models I tried (linear attention variants, single-layer only, fixing memory tokens etc) but none of these were successful.
- `src/models/q_table.py`: A simple Q-table model for comparison.
- `src/models/dqn.py`: A DQN model for comparison.
- `src/models/rnn.py`: LSTM model for comparison. Follows the same pretraining procedure as the transformer.
- The `src/agents` and `src/evals` folders contain essentially the same code and structure as the original DPT repo.

## Creating datasets
To create the training dataset, I run `python make_tree_seeds.py && python collect_data.py`. The first script records which random seeds correspond to which randomly generated tree structure. These are then partitioned into training/validation sets and saved. The second scripts uses that information to run each environment to collect the pretraining data. Typically, the `data_collection.yaml` config is used for this, with the default settings of `configs/envs/cntree.yaml`.

To create the testing dataset, I run `python collect_data.py`, and modify `data_collection.yaml` to set `n_envs: 1000`, `branching_prob: 1.0`, and `horizon: 1600` (I really only use up to context length 1000 in testing though). This is the same as the pretraining data collection, but it only runs 1000 episodes for each tree structure. I don't use `make_tree_seeds.py` for testing because there's only one tree structure at test.

There's also a `collect_data_h5.py` script that's similar to `collect_data.py`, but it saves the data in HDF5 format. This was useful when I was trying to collect large datasets and ran out of memory, but I didn't see much performance improvement from scaling the training dataset up, and have not needed this script since.

## Training
Training is done with `python train.py`. The `training.yaml` config is used for this.

## Testing
To make the main learning curves, I use `python eval_offline.py`. There's also `python eval.py` which follows the original repo more and includes the online evaluation stuff. Similarly, the main Q-learning eval script is `eval_qlearning_offline.py`, which can be used for either the DQN or the Q-table models.

## Notebooks
All figures are made in the `notebooks` folder. The main representation metric I settled on is kernel alignment, but there's a lot of other things I tried still in the repo.