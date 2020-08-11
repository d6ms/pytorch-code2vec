
import os

import torch

# train configurations
EMBEDDING_DIM = 128
DROPOUT = 0.25
BATCH_SIZE = 1024
CHUNK_SIZE = 10
MAX_LENGTH = 200
SAVE_EVERY = 1000

# predict configurations
JAR_PATH = 'JavaExtractor-0.0.1-SNAPSHOT.jar'
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = f'{BASE_PATH}/data/java14m'
MODEL_PATH = f'{BASE_PATH}/models'
LOG_PATH = f'{BASE_PATH}/logs'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
