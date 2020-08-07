import pickle

from utils import create_dirs, fix_seeds
fix_seeds(1234)
from train import train
from data import BatchDataLoader, Vocabulary


if __name__ == '__main__':
    # args, parser = parse_args()

    create_dirs()

    with open('data/java14m/java14m.dict.c2v', 'rb') as f:
        word2count = pickle.load(f)
        path2count = pickle.load(f)
        label2count = pickle.load(f)
        # n_training_examples = pickle.load(f)
    word_vocab = Vocabulary(word2count.keys())
    path_vocab = Vocabulary(path2count.keys())
    label_vocab = Vocabulary(label2count.keys())

    loader = BatchDataLoader('data/java14m/java14m.train.c2v', word_vocab, path_vocab, label_vocab)
    i = 0
    for name, x_s, path, x_t, mask in loader:
        print(name, x_s, path, x_t, mask)
        i += 1
        if i > 0:
            break

    train()