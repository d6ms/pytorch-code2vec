from datetime import datetime

import numpy as np
import pickle
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
matplotlib.use('Agg')

import config
from models import Code2Vec
from data import BatchDataLoader, Vocabulary


def train(epochs, lr=0.001):
    # prepare dataloaders
    with open(f'{config.DATA_PATH}/java14m.dict.c2v', 'rb') as f:
        word2count = pickle.load(f)
        path2count = pickle.load(f)
        label2count = pickle.load(f)
        # n_training_examples = pickle.load(f)
    word_vocab = Vocabulary(word2count.keys())
    path_vocab = Vocabulary(path2count.keys())
    label_vocab = Vocabulary(label2count.keys())
    trainloader = BatchDataLoader(f'{config.DATA_PATH}/java14m.train.c2v', word_vocab, path_vocab, label_vocab)
    evalloader = BatchDataLoader(f'{config.DATA_PATH}/java14m.test.c2v', word_vocab, path_vocab, label_vocab)

    # train settings
    model = Code2Vec(len(word_vocab), len(path_vocab), config.EMBEDDING_DIM, len(label_vocab), config.DROPOUT).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss().to(config.DEVICE)

    # train
    history = {'eval_loss': list(), 'eval_acc': list()}
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, loss_fn, trainloader, epoch)
        evaluate(model, history, loss_fn, evalloader, epoch)

        # save model and history
        if epoch == 1 or history['eval_acc'][-1] > max(history['eval_acc'][:-1]):
            torch.save(model.state_dict(), f'{config.MODEL_PATH}/code2vec.ckpt')
        save_history(history)

def train_epoch(model, optimizer, loss_fn, dataloader, epoch_idx):
    model.train()
    for i, (label, x_s, path, x_t, mask) in enumerate(dataloader, 1):
        # label: 正解ラベルのメソッド名を示すindex (batch_size,)
        # x_s : ASTコンテキストの始点ラベルを示すindex (batch_size, max_length)
        # path: ASTパスのhashラベルを示すindex (batch_size, max_length)
        # x_t : ASTコンテキストの終点ラベルを示すindex (batch_size, max_length)
        # mask: axis-1 のサイズを max_length に合わせるためにパディングした箇所は 1 (batch_size, max_length)
        label, x_s, path, x_t = label.to(config.DEVICE), x_s.to(config.DEVICE), path.to(config.DEVICE), x_t.to(config.DEVICE)

        optimizer.zero_grad()
        out = model(x_s, path, x_t)
        loss = loss_fn(out, label)

        loss.backward()
        optimizer.step()

        print(f'{datetime.now()} [epoch {epoch_idx} batch {i}] loss: {loss.item()}')

def evaluate(model, history, loss_fn, dataloader, epoch_idx):
    model.eval()

    total_loss, total_correct, data_cnt = 0, 0, 0
    with torch.no_grad():
        for i, (label, x_s, path, x_t, mask) in enumerate(dataloader, 1):
            label, x_s, path, x_t = label.to(config.DEVICE), x_s.to(config.DEVICE), path.to(config.DEVICE), x_t.to(config.DEVICE)

            out = model(x_s, path, x_t)
            loss = loss_fn(out, label)

            predicted = out.max(1, keepdim=True)[1]
            n_correct = predicted.eq(label.view_as(predicted)).sum()

            data_cnt += label.shape[0]
            total_loss += loss.item() * label.shape[0]
            total_correct += n_correct

    history['eval_loss'].append(total_loss / data_cnt)
    history['eval_acc'].append(total_correct / data_cnt)
    print(f'{datetime.now()} [epoch {epoch_idx} eval] loss: {total_loss / data_cnt}, accuracy: {total_correct / data_cnt}')

def save_history(history):
    for metric, values in history.items():
        # save raw data
        with open(f'{config.LOG_PATH}/{metric}.data', mode='w') as f:
            data = ','.join([str(v) for v in values])
            f.write(data)
        # save graph
        x, y = np.linspace(1, len(values), len(values)), np.array(values)
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.title(metric)
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.savefig(f'{config.LOG_PATH}/{metric}.png')