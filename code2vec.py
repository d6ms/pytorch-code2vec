from utils import create_dirs, fix_seeds
fix_seeds(1234)
from train import train


if __name__ == '__main__':
    # args, parser = parse_args()

    create_dirs()

    train(1)