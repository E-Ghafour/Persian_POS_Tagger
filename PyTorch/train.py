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
                      pad_len=100,
                      word2vec = word2vec,
                      padded_x_path='padded_x60.pkl',
                      padded_y_path='padded_y.pkl'
                      )





