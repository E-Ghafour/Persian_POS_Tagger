from my_dataset import Dataset_POS
from my_lstm_model import Model_LSTM_POS
import pickle
import numpy as np
import pandas as pd 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set to the index of the desired GPU
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from configparser import ConfigParser
import pickle
from torch.utils.data import Dataset
import itertools
from tqdm import tqdm
from hazm import WordEmbedding
import torch.optim.lr_scheduler as lr_scheduler


# training info
device = 'cuda'
torch.device('cuda')

# sents_num = 344741

sents_num = 3447
labels_num = 23

pad_len = 60
test_size = int(0.05 * sents_num)
train_size = sents_num - test_size
batch_size = 512
input_size = 300
hidden_size = 64
output_size = 24
num_layers = 2
bidirectional = True
inner_dropout = 0.5
dropout = 0.5
lr = 0.005
epochs = 20



word2vec = WordEmbedding(model_path='cc.fa.300.bin', model_type='fasttext')


dataset = Dataset_POS(tokenize_data_path='peykare.pkl',
                      pad_len=pad_len,
                      word2vec = word2vec,
                      padded_x_path='padded_x60.pkl',
                      padded_y_path='padded_y.pkl'
                      )


train_dataset, test_dataset = random_split(dataset=dataset,
                                                 lengths=[train_size, test_size]
                                                 )



train_dataLoader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=5
                              )
test_dataLoader = DataLoader(dataset=test_dataset,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   num_workers=5
                                   )



vectors = word2vec.get_vectors()
vectors = np.append(vectors, [np.zeros_like(vectors[0])], axis=0)

model = Model_LSTM_POS(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=output_size,
                       num_layers=num_layers,
                       bidirectional=bidirectional,
                       inner_dropout=inner_dropout,
                       dropout=dropout,
                       vectors=vectors).to(device)


sample = next(iter(train_dataLoader))



loss_fn = nn.BCEWithLogitsLoss()
loss_fn = loss_fn.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)



def train_epoch():
    model.train()
    num_correct = 0
    num_tokens = 0
    for batch, (X, y) in enumerate(train_dataLoader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        padding_mask = (X < 2000000).int()
        target_masked = y * padding_mask.unsqueeze(-1)
        pred = model(X)
        loss = loss_fn(torch.masked_select(pred, target_masked.bool()), torch.masked_select(target_masked, target_masked.bool()))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc
        mask = y.argmax(dim=-1) < labels_num
        num_correct += (torch.eq(y.argmax(dim=-1), pred.argmax(dim=-1)) & mask).sum().item()
        num_tokens += mask.sum().item()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{sents_num:>5d}]")
    print(f'train accuracy: {(100*num_correct/num_tokens):>0.01f}')





def test_accuracy():
    model.eval()
    num_tokens = 0
    num_correct = 0
    with torch.no_grad():
        for x, y in test_dataLoader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mask = y.argmax(dim=-1) < labels_num
            num_correct += (torch.eq(y.argmax(dim=-1), pred.argmax(dim=-1)) & mask).sum().item()
            num_tokens += mask.sum().item()
    return num_correct/num_tokens



best_acc = 0.95
for epoch in range(100):
    print(f'epoch: {epoch} of {epochs}')
    train_epoch()
    acc = test_accuracy()
    print(f'the test accuracy: {acc}')
    if(best_acc < acc):
        print('saving_model...')
        best_acc = acc
        torch.save(model, 'best_acc.model')
    print('------------------------------------------')
