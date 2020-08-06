import os
import random

import numpy as np
import torch

import config


def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs():
    for dir in [config.MODEL_PATH, config.LOG_PATH]:
        if not os.path.exists(dir):
            os.makedirs(dir)