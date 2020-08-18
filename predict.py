import subprocess
import random
from typing import List

import torch

import config
from models import Code2Vec
from data import Vocabulary, load_vocabularies


def predict(filename: str, model_path: str):
    word_vocab, path_vocab, label_vocab = load_vocabularies()
    model = Code2Vec(len(word_vocab), len(path_vocab), config.EMBEDDING_DIM, len(label_vocab), config.DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

    for _, ctxs in extract(filename):
        x_s, path, x_t = tensorize(ctxs, word_vocab, path_vocab)
        out, code_vector = model(x_s, path, x_t)
        print('code vector', code_vector.squeeze(0))
        for idx in torch.topk(out.squeeze(0), k=6)[1]:
            if idx == 0:
                continue
            predicted = label_vocab.lookup_word(int(idx))
            print(predicted)

def extract(filename):
    command = ['java', '-cp', config.JAR_PATH, 'JavaExtractor.App', '--max_path_length', str(config.MAX_PATH_LENGTH), '--max_path_width', str(config.MAX_PATH_WIDTH), '--file', filename, '--no_hash']
    out, err = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    methods = out.decode().splitlines()
    for method in methods:
        method_name, *ctxs = method.split(' ')
        yield method_name, ctxs

def tensorize(ctxs: List[str], word_vocab: Vocabulary, path_vocab: Vocabulary):
    if len(ctxs) > config.MAX_LENGTH:
        ctxs = random.sample(ctxs, config.MAX_LENGTH)
    x_s, path, x_t = [0] * config.MAX_LENGTH, [0] * config.MAX_LENGTH, [0] * config.MAX_LENGTH
    for i in range(config.MAX_LENGTH):
        if i < len(ctxs):
            s, p, t = ctxs[i].split(',')
            p = str(java_string_hashcode(p))
        else:
            s, p, t = '<pad>', '<pad>', '<pad>'
        x_s[i] = word_vocab.lookup_idx(s)
        path[i] = path_vocab.lookup_idx(p)
        x_t[i] = word_vocab.lookup_idx(t)
    x_s, path, x_t = torch.LongTensor(x_s)[None, :], torch.LongTensor(path)[None, :], torch.LongTensor(x_t)[None, :]
    return x_s, path, x_t

def java_string_hashcode(s):
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
