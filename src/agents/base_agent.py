import itertools

import numpy as np
import scipy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def set_batch(self, batch):
        self.batch = batch

    def set_batch_numpy_vec(self, batch):
        self.set_batch(batch)

    def set_env(self, env):
        self.env = env
