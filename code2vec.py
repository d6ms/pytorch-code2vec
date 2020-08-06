from utils import create_dirs, fix_seeds
from train import train


if __name__ == '__main__':
    # args, parser = parse_args()

    fix_seeds(1234)
    create_dirs()

    train()