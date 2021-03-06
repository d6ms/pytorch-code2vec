import random
from collections import deque
from typing import Iterable

import torch
import pickle

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

    def __len__(self):
        return len(self.word2idx)


# PyTorch の Dataset, DataLoader を使う場合はデータへのランダムアクセスを行う必要があるが、
# 今回使用するデータは非常に大きなテキストファイルであり、ランダムアクセスが実際上不可能であるため、
# シーケンシャルアクセスでバッチを作成する自作の DataLoader を使用する。
class Code2VecBatchDataLoader(object):

    def __init__(self, data_path: str, word_vocab: Vocabulary, path_vocab: Vocabulary, label_vocab: Vocabulary):
        self.data_path = data_path
        self.word_vocab = word_vocab
        self.path_vocab = path_vocab
        self.label_vocab = label_vocab
        self.chunks = deque()
        with open(data_path, mode='r') as f:
            self.n_data = sum(1 for _ in f)

    def __len__(self):
        return self.n_data // config.BATCH_SIZE

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
        if len(self.chunks) == 0:
            self.__load_chunks()
        return self.chunks.popleft()
    
    def __load_chunks(self):
        samples = []
        for _ in range(config.BATCH_SIZE * config.CHUNK_SIZE):
            line = self.data.__next__()

            # parse line
            method_name, *ast_ctxs = line.split(' ')
            ast_ctxs = [ctx.split(',') for ctx in ast_ctxs if ctx != '' and ctx != '\n']
            n_ctxs = len(ast_ctxs)
            assert n_ctxs <= config.MAX_LENGTH

            # AST コンテキストの MAX_LENGTH に満たないサイズを padding
            ast_ctxs += [['<pad>', '<pad>', '<pad>']] * (config.MAX_LENGTH - n_ctxs)

            samples.append((method_name, ast_ctxs, n_ctxs))
        
        # batch_size * chunk_size 個のデータをfetchしてからshuffleする
        random.shuffle(samples)

        # batch_size ごとの chunk に分割して queue に追加
        num_chunks = config.CHUNK_SIZE
        if len(samples) < config.BATCH_SIZE * config.CHUNK_SIZE:
            num_chunks -= 1
        for i in range(num_chunks):
            chunk = samples[i * config.BATCH_SIZE: (i + 1) * config.BATCH_SIZE]
            self.chunks.append(chunk)

    def __tensorize(self, samples):
        label = torch.zeros(config.BATCH_SIZE).long()
        x_s = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()
        path = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()
        x_t = torch.zeros((config.BATCH_SIZE, config.MAX_LENGTH)).long()

        for i, (method_name, ctxs, n_ctxs) in enumerate(samples):
            label[i] = self.label_vocab.lookup_idx(method_name)
            tmp_x_s, tmp_path, tmp_x_t = zip(*[(
                self.word_vocab.lookup_idx(s),
                self.path_vocab.lookup_idx(p),
                self.word_vocab.lookup_idx(t)
            ) for s, p, t in ctxs])
            x_s[i, :], path[i, :], x_t[i, :] = torch.LongTensor(tmp_x_s), torch.LongTensor(tmp_path), torch.LongTensor(tmp_x_t)

        return label, x_s, path, x_t


class Code2SeqBatchDataLoader(object):

    def __init__(self, data_path: str, word_vocab: Vocabulary, path_vocab: Vocabulary, label_vocab: Vocabulary):
        self.data_path = data_path
        self.word_vocab = word_vocab
        self.path_vocab = path_vocab
        self.label_vocab = label_vocab
        self.chunks = deque()
        with open(data_path, mode='r') as f:
            self.n_data = sum(1 for _ in f)

    def __len__(self):
        return self.n_data // config.BATCH_SIZE

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
        if len(self.chunks) == 0:
            self.__load_chunks()
        return self.chunks.popleft()
    
    def __load_chunks(self):
        def pad(arr, length):
            arr = arr[:length]
            arr += ['<pad>'] * (length - len(arr))
            return arr

        samples = []
        for _ in range(config.BATCH_SIZE * config.CHUNK_SIZE):
            line = self.data.__next__()

            # parse line
            method_name, *ctxs = line.split(' ')
            ctxs = [ctx for ctx in ctxs if ctx != '' and ctx != '\n']
            if len(ctxs) > config.MAX_LENGTH:
                ctxs = random.sample(ctxs, config.MAX_LENGTH)
            ast_ctxs = []
            for ctx in ctxs:
                x_s, path, x_t = ctx.split(',')
                ast_ctxs.append([
                    pad(x_s.split('|'), config.MAX_NAME_PARTS),
                    pad(path.split('|'), config.MAX_PATH_LENGTH),
                    pad(x_t.split('|'), config.MAX_NAME_PARTS)
                ])
            n_ctxs = len(ast_ctxs)
            assert n_ctxs <= config.MAX_LENGTH

            # AST コンテキストの MAX_LENGTH に満たないサイズを padding
            ast_ctxs += [[
                ['<pad>'] * config.MAX_NAME_PARTS,
                ['<pad>'] * config.MAX_PATH_LENGTH,
                ['<pad>'] * config.MAX_NAME_PARTS
            ]] * (config.MAX_LENGTH - n_ctxs)

            samples.append((method_name, ast_ctxs, n_ctxs))
        
        # batch_size * chunk_size 個のデータをfetchしてからshuffleする
        random.shuffle(samples)

        # batch_size ごとの chunk に分割して queue に追加
        num_chunks = config.CHUNK_SIZE
        if len(samples) < config.BATCH_SIZE * config.CHUNK_SIZE:
            num_chunks -= 1
        for i in range(num_chunks):
            chunk = samples[i * config.BATCH_SIZE: (i + 1) * config.BATCH_SIZE]
            self.chunks.append(chunk)

    def __tensorize(self, samples):
        x_s, path, x_t, label = [], [], [], []
        for method_name, ctxs, n_ctxs in samples:
            label.append(self.label_vocab.lookup_idx(method_name))
            tmp_x_s, tmp_path, tmp_x_t = [], [], []
            for ctx in ctxs:
                tmp_x_s.append([self.word_vocab.lookup_idx(s) for s in ctx[0]])
                tmp_path.append([self.path_vocab.lookup_idx(p) for p in ctx[1]])
                tmp_x_t.append([self.word_vocab.lookup_idx(t) for t in ctx[2]])
            x_s.append(tmp_x_s)
            path.append(tmp_path)
            x_t.append(tmp_x_t)
        label, x_s, path, x_t = torch.LongTensor(label), torch.LongTensor(x_s), torch.LongTensor(path), torch.LongTensor(x_t)
        return label, x_s, path, x_t

def load_vocabularies(vocab_path):
    with open(vocab_path, 'rb') as f:
        word2count = pickle.load(f)
        path2count = pickle.load(f)
        label2count = pickle.load(f)
    word_vocab = Vocabulary(word2count.keys())
    path_vocab = Vocabulary(path2count.keys())
    label_vocab = Vocabulary(label2count.keys())
    return word_vocab, path_vocab, label_vocab