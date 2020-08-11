import sys
import logging
import traceback
from argparse import ArgumentParser

import config
from utils import create_dirs, fix_seeds, configure_logger
fix_seeds(1234)
from train import train
from predict import predict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train model')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-p', '--predict', action='store_true', help='embed code and predict method name')
    parser.add_argument('--file', default='Test.java', type=str)
    parser.add_argument('--model', default='models/code2vec.ckpt', type=str)
    args = parser.parse_args()
    return args, parser

if __name__ == '__main__':
    args, parser = parse_args()
    config.BATCH_SIZE = args.batch_size

    create_dirs()
    configure_logger()

    if not any([args.train, args.predict]):
        parser.print_help()
        exit(1)

    try:
        if args.train:
            train(args.epochs, lr=args.lr)
        elif args.predict:
            predict(args.file, args.model)
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        logging.exception(e)