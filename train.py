import logging

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
matplotlib.use('Agg')

import config
from models import Code2Vec
from data import BatchDataLoader, load_vocabularies


def train(epochs, lr=0.001):
    logging.info('start training')

    # prepare dataloaders
    word_vocab, path_vocab, label_vocab = load_vocabularies()
    trainloader = BatchDataLoader(f'{config.DATA_PATH}/java-large.train.c2v', word_vocab, path_vocab, label_vocab)
    evalloader = BatchDataLoader(f'{config.DATA_PATH}/java-large.val.c2v', word_vocab, path_vocab, label_vocab)
    logging.info(f'trains over {len(trainloader)} batches, evaluates over {len(evalloader)} batches')

    # train settings
    model = Code2Vec(len(word_vocab), len(path_vocab), len(label_vocab), config.EMBEDDING_DIM, config.EMBEDDING_DIM * 3, config.DROPOUT).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss().to(config.DEVICE)

    # train
    history = {'eval_loss': list(), 'eval_acc': list(), 'eval_precision': list(), 'eval_recall': list(), 'eval_f1': list()}
    def after_batch(epoch, batch_idx):
        if batch_idx % config.SAVE_EVERY == 0 or batch_idx == len(trainloader):
            evaluate(model, history, loss_fn, evalloader, epoch, label_vocab)
            if len(history['eval_f1']) == 1 or history['eval_f1'][-1] > max(history['eval_f1'][:-1]):
                torch.save(model.encoder.state_dict(), f'{config.MODEL_PATH}/encoder.ckpt')
                torch.save(model.state_dict(), f'{config.MODEL_PATH}/code2vec.ckpt')
                logging.info(f'[epoch {epoch}] model saved')
            save_history(history)
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, loss_fn, trainloader, epoch, label_vocab, after_batch_callback=after_batch)

def train_epoch(model, optimizer, loss_fn, dataloader, epoch_idx, label_vocab, after_batch_callback=None):
    for i, (label, x_s, path, x_t) in enumerate(dataloader, 1):
        # label: 正解ラベルのメソッド名を示すindex (batch_size,)
        # x_s : ASTコンテキストの始点ラベルを示すindex (batch_size, max_length)
        # path: ASTパスのhashラベルを示すindex (batch_size, max_length)
        # x_t : ASTコンテキストの終点ラベルを示すindex (batch_size, max_length)
        label, x_s, path, x_t = label.to(config.DEVICE), x_s.to(config.DEVICE), path.to(config.DEVICE), x_t.to(config.DEVICE)

        model.train()
        optimizer.zero_grad()

        out, _ = model(x_s, path, x_t)
        loss = loss_fn(out, label)
        accuracy = compute_accuracy(out, label)
        precision, recall, f1 = compute_f1(out, label, label_vocab)

        loss.backward()
        optimizer.step()

        logging.info(f'[epoch {epoch_idx} batch {i}] loss: {loss.item()}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}')

        if after_batch_callback is not None:
            after_batch_callback(epoch_idx, i)

def evaluate(model, history, loss_fn, dataloader, epoch_idx, label_vocab):
    model.eval()

    total_loss, total_acc, total_precision, total_recall, total_f1, batch_cnt, data_cnt = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, (label, x_s, path, x_t) in enumerate(dataloader, 1):
            label, x_s, path, x_t = label.to(config.DEVICE), x_s.to(config.DEVICE), path.to(config.DEVICE), x_t.to(config.DEVICE)

            out, _ = model(x_s, path, x_t)
            loss = loss_fn(out, label)

            data_cnt += label.shape[0]
            total_loss += loss.item() * label.shape[0]
            total_acc += compute_accuracy(out, label)
            precision, recall, f1 = compute_f1(out, label, label_vocab)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            batch_cnt += 1

    history['eval_loss'].append(total_loss / data_cnt)
    history['eval_acc'].append(total_acc / batch_cnt)
    history['eval_precision'].append(total_precision / batch_cnt)
    history['eval_recall'].append(total_recall / batch_cnt)
    history['eval_f1'].append(total_f1 / batch_cnt)
    logging.info(f'[epoch {epoch_idx} eval] loss: {total_loss / data_cnt}, accuracy: {total_acc / batch_cnt}, precision: {total_precision / batch_cnt}, recall: {total_recall / batch_cnt}, f1: {total_f1 / batch_cnt}')

def compute_accuracy(fx, y):
    pred_idxs = fx.max(1, keepdim=True)[1]
    correct = pred_idxs.eq(y.view_as(pred_idxs)).sum()
    acc = correct.float() / pred_idxs.shape[0]
    return acc

def compute_f1(fx, y, label_vocab):
    pred_idxs = fx.max(1, keepdim=True)[1]
    pred_names = [label_vocab.lookup_word(i.item()) for i in pred_idxs]
    original_names = [label_vocab.lookup_word(i.item()) for i in y]
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(pred_names, original_names):
        predicted_subtokens = p.split('|')
        original_subtokens = o.split('|')
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1

def correct_count(out, label):
    predicted = out.max(1, keepdim=True)[1]
    n_correct = predicted.eq(label.view_as(predicted)).sum()
    return n_correct

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