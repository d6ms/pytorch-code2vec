import random
from typing import Iterable

import torch

import config


class Vocabulary(object):

    def __init__(self, words: Iterable[str]):
        self.word2idx = {word: i for i, word in enumerate(['<unk>', '<pad>', *words])}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def lookup_idx(self, word):
        if word not in self.word2idx:
            return 0
        else:
            return self.word2idx[word]

    def lookup_word(self, idx):
        if idx not in self.idx2word:
            return '<unk>'
        else:
            return self.idx2word[idx]


# PyTorch の Dataset, DataLoader を使う場合はデータへのランダムアクセスを行う必要があるが、
# 今回使用するデータは非常に大きなテキストファイルであり、ランダムアクセスが実際上不可能であるため、
# シーケンシャルアクセスでバッチを作成する自作の DataLoader を使用する。
class BatchDataLoader(object):

    def __init__(self, data_path: str, word_vocab: Vocabulary, path_vocab: Vocabulary, label_vocab: Vocabulary):
        self.data_path = data_path
        self.word_vocab = word_vocab
        self.path_vocab = path_vocab
        self.label_vocab = label_vocab

    def __iter__(self):
        self.data = self.__generator()
        return self

    def __next__(self):
        samples = self.__next_batch_samples()
        return self.__tensorize(samples)

    def __generator(self):
        with open(self.data_path, mode='r') as f:
            for line in f:
                yield line.rstrip()

    def __next_batch_samples(self):
        samples = []
        for _ in range(config.BATCH_SIZE):
            line = self.data.__next__()

            # parse line
            method_name, *ast_ctxs = line.split(' ')
            ast_ctxs = [ctx.split(',') for ctx in ast_ctxs if ctx != '' and ctx != '\n']
            n_ctxs = len(ast_ctxs)
            assert n_ctxs <= config.MAX_LENGTH

            # AST コンテキストの MAX_LENGTH に満たないサイズを padding
            ast_ctxs += [['<pad>', '<pad>', '<pad>']] * (config.MAX_LENGTH - n_ctxs)

            samples.append((method_name, ast_ctxs, n_ctxs))

        random.shuffle(samples)
        return samples

    def __tensorize(self, samples):
        name = torch.zeros(config.BATCH_SIZE).long()
        x_s = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()
        path = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()
        x_t = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()
        mask = torch.ones((config.BATCH_SIZE, config.MAX_LENGTH)).float()

        for i, (method_name, ctxs, n_ctxs) in enumerate(samples):
            name[i] = self.label_vocab.lookup_idx(method_name)
            tmp_x_s, tmp_path, tmp_x_t = zip(*[(
                self.word_vocab.lookup_idx(s),
                self.path_vocab.lookup_idx(p),
                self.word_vocab.lookup_idx(t)
            ) for s, p, t in ctxs])
            x_s[i, :], path[i, :], x_t[i, :] = torch.LongTensor(tmp_x_s), torch.LongTensor(tmp_path), torch.LongTensor(tmp_x_t)
            mask[i, n_ctxs:] = 0

        return name, x_s, path, x_t, mask
