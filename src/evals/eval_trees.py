import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.agents.agent import (
    TransformerAgent,
    OptPolicy,
)
from src.envs.trees import (
    TreeEnv,
    TreeEnvVec,
)
from src.utils import convert_to_tensor
from src.evals.eval_darkroom import EvalDarkroom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalTrees(EvalDarkroom):
    def __init__(self):
        pass

    def create_env(self, config, goal, i_eval):
        _config = {
            'max_layers': config['max_layers'],
            'initialization_seed': config['initialization_seed'][i_eval],
            'horizon': config['horizon'],
            'branching_prob': config['branching_prob']}
        return TreeEnv(**_config)

    def create_vec_env(self, envs):
        return TreeEnvVec(envs)
    
    def offline(self, eval_trajs, model, config, plot=False):
        """Runs each episode separately with offline context, after calculating tree-specific metrics."""

        config['initialization_seed'] = [
            eval_trajs[i_eval]['initialization_seed'] for i_eval in range(len(eval_trajs))]
        return super().offline(eval_trajs, model, config, plot)

