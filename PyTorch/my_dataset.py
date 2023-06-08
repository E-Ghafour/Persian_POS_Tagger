import torch
from torch.utils.data import Dataset
import pickle
import itertools
import numpy as np


class Dataset_POS(Dataset):
    def __init__(self, tokenize_data_path, pad_len, word2vec):
        self.pad_len = pad_len
        with open(tokenize_data_path, 'rb') as ff:
            sents = pickle.load(ff)
        self.tokens = [[word for word, _ in sent]for sent in sents]
        self.labels = [[label for _, label in sent]for sent in sents]
        self.label_to_id = self.__create_label_dict()
        self.vocab_to_id = word2vec.get_vocab_to_index()
        self.vocabs = word2vec.get_vocabs()
        pad_label_one_hot = [0] * (self.num_labels + 1)
        pad_label_one_hot[-1] = 1
        self.pad_label_one_hot = pad_label_one_hot

    def __len__(self):
        return len(self.tokenize_data)

    def __create_label_dict(self):
        flatten_label = list(itertools.chain(*self.labels))
        unique_labels = np.unique(flatten_label).tolist()
        self.num_labels = len(unique_labels)
        label_dict = {label:i for i, label in enumerate(unique_labels)}
        return label_dict

    def __convert_label_to_onehot(self, label: str):
        one_hot = [0] * (self.num_labels + 1)
        one_hot[self.label_to_id[label]] = 1
        return one_hot
    
    def convert_labels_to_onehot(self, labels: list):
        return [self.__convert_label_to_onehot(label) for label in labels]
        
    def __getitem__(self, idx):
        x = self.tokens[idx]
        y = self.labels[idx]

        x_indices = [self.vocab_to_id[token] if token in self.vocabs else 'unk' for token in x]
        x_padded_indices = x_indices[:self.pad_len] + [0] * (self.pad_len - len(x_indices[:self.pad_len]))

        y = self.convert_labels_to_onehot(y)
        y_padded_indices = y[:self.pad_len] + [self.pad_label_one_hot] * (self.pad_len - len(y[:self.pad_len]))

        return torch.tensor(x_padded_indices), torch.tensor(y_padded_indices)