import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.agents.agent import (
    TransformerAgent,
    OptPolicy,
)
from src.envs.maze_env import (
    MazeEnv,
    MazeEnvVec,
)
from src.utils import convert_to_tensor
from src.evals.eval_darkroom import EvalDarkroom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalMaze(EvalDarkroom):
    def __init__(self):
        pass

    def create_env(self, config, goal, i_eval):
        layers = config['layers']
        horizon = config['horizon']
        return MazeEnv(layers, goal, horizon)


    def create_vec_env(self, envs):
        return MazeEnvVec(envs)

