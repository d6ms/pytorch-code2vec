
import os

import torch

EMBEDDING_DIM = 128
DROPOUT = 0.25
BATCH_SIZE = 1024
MAX_LENGTH = 200

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = f'{BASE_PATH}/data/java14m'
MODEL_PATH = f'{BASE_PATH}/models'
LOG_PATH = f'{BASE_PATH}/logs'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
