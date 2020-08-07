from argparse import ArgumentParser

import config
from utils import create_dirs, fix_seeds
fix_seeds(1234)
from train import train


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train model')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    return args, parser

if __name__ == '__main__':
    args, parser = parse_args()
    config.BATCH_SIZE = args.batch_size

    create_dirs()

    if args.train:
        train(args.epochs, lr=args.lr)
    else:
        parser.print_help()