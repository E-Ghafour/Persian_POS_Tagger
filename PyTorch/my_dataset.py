import torch
from torch.utils.data import Dataset
import pickle
import itertools
import numpy as np


class Dataset_POS(Dataset):
    def __init__(self, tokenize_data_path, pad_len, word2vec):
        self.max_len = pad_len
        with open(tokenize_data_path, 'rb') as ff:
            sents = pickle.load(ff)
        self.tokens = [[word for word, _ in sent]for sent in sents]
        self.labels = [[label for _, label in sent]for sent in sents]
        flatten_label = list(itertools.chain(*self.labels))
        self.unique_labels = np.unique(flatten_label).tolist()
        self.num_labels = len(self.unique_labels)
        self.vocab_to_id = word2vec.get_vocab_to_id()
        self.vocabs = word2vec.get_vocabs()

    def __len__(self):
        return len(self.tokenize_data)
    
    def __convert_label_to_onehot(self, label: str):
        one_hot = [0] * self.num_labels
        one_hot[self.unique_labels.index(label)] = 1
        return one_hot
    
    def convert_labels_to_onehot(self, labels: list):
        return [self.__convert_label_to_onehot(label) for label in labels]
        
    def __getitem__(self, idx):
        x = self.tokens[idx]
        y = self.tokens[idx]

        x_indices = [self.vocab_to_id(token) if token in self.vocabs else 'unk' for token in x]
        x_padded_indices = x_indices[:self.max_len] + [0] * (self.max_len - len(x_indices[:self.max_len]))

        y = self.convert_labels_to_onehot(y)

        return torch.tensor(x_padded_indices), y