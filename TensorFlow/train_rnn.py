import nltk
from hazm import *
from nltk.corpus import treebank, conll2000, brown
import pickle
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, TimeDistributed, Dense, LSTM, GRU
import itertools

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.model_selection import train_test_split

with open('../data.pkl', 'rb') as ff:
    tagged_sentences = pickle.load(ff)


X = [] # store input sequence
Y = [] # store output sequencefor sentence in tagged_sentences:

for sentence in tqdm(tagged_sentences):
    X_sentence = []
    Y_sentence = []
    for entity in sentence: 
        X_sentence.append(entity[0]) # entity[0] contains the word
        Y_sentence.append(entity[1]) # entity[1] contains corresponding tag
 
    X.append(X_sentence)
    Y.append(Y_sentence)

num_words = len(set([word.lower() for sentence in X for word in sentence]))
num_tags   = len(set([word.lower() for sentence in Y for word in sentence]))
print("Total number of tagged sentences: {}".format(len(X)))
print("Vocabulary size: {}".format(num_words))
print("Total number of tags: {}".format(num_tags))


# let’s look at first data point
# this is one data point that will be fed to the RNN
print('sample X: ', X[0], '\n')
print('sample Y: ', Y[0], '\n')

# In this many-to-many problem, the length of each input and output sequence must be the same.
# Since each word is tagged, it’s important to make sure that the length of input sequence equals the output sequenceprint(“Length of first input sequence : {}”.format(len(X[0])))
print('Length of first output sequence : {}'.format(len(Y[0])))

we = WordEmbedding(model_path='/home/roshan/ebi/word_embedding/resources/cc.fa.300.bin',
                   model_type='fasttext')

vocab_to_index = we.get_vocab_to_index()
vocabs = we.get_vocabs()
vectors = we.get_vectors()

arr = list(vectors)
arr.insert(0, np.zeros(300, ))
vectors = np.array(arr)

# encode X
X_encoded = []
for sent in tqdm(X):
    tmp_list = []
    for word in sent:
        tmp_list.append(vocab_to_index.get(word, 0))
    X_encoded.append(tmp_list)


# encode Y
def create_dict(Y):
    labels = []
    for sent in tqdm(Y):
        for label in sent:
            labels.append(label)
    unique_labels = np.unique(labels).tolist()
    unique_labels.insert(0, 'PAD')
    label_dict = {}
    for i in range(len(unique_labels)):
        label_dict[unique_labels[i]] = i
    return label_dict

label_dict = create_dict(Y)
Y_encoded = []
for sent in tqdm(Y):
    tmp_list = []
    for label in sent:
        tmp_list.append(label_dict[label])
    Y_encoded.append(tmp_list)

id2label = {}
for label, id in label_dict.items():
    id2label[id] = label


# look at first encoded data point
print("** Raw data point **", "\n", "-"*100, "\n")
print('X: ', X[0], '\n')
print('Y: ', Y[0], '\n')
print()
print("** Encoded data point **", "\n", "-"*100, "\n")
print('X: ', X_encoded[0], '\n')
print('Y: ', Y_encoded[0], '\n')


MAX_SEQ_LENGTH = 60
X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

# print the first sequence
print(X_padded[0], "\n"*3)
print(Y_padded[0])

EMBEDDING_SIZE  = 300
VOCABULARY_SIZE = len(vectors)
embedding_weights = vectors
word2id = we.get_vocab_to_index()


Y = to_categorical(Y_padded)
X = X_padded
NUM_CLASSES = len(Y[0][0])


# create architecture
rnn_model = Sequential()
# create embedding layer — usually the first layer in text problems
# vocabulary size — number of unique words in data
rnn_model.add(Embedding(input_dim = VOCABULARY_SIZE, 
# length of vector with which each word is represented
 output_dim = EMBEDDING_SIZE, 
# length of input sequence
 input_length = MAX_SEQ_LENGTH, 
# False — don’t update the embeddings
 trainable = False 
))
# add an RNN layer which contains 64 RNN cells
# True — return whole sequence; False — return single output of the end of the sequence
rnn_model.add(SimpleRNN(64, 
 return_sequences=True
))
# add time distributed (output at each sequence) layer
rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#compile model
rnn_model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   =  ['acc'])
# check summary of the model
rnn_model.summary()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=10)


#fit model
rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128,
                            epochs=30, validation_data=(X_test, Y_test))


Y_test = Y_test.argmax(axis=-1)
Y_test_flatten_tmp = Y_test.flatten()
Y_test_flatten = Y_test_flatten_tmp[Y_test_flatten_tmp > 0]
Y_test_flatten = [id2label[id] for id in Y_test_flatten]


Y_pred = rnn_model.predict(X_test)
Y_pred = Y_pred.argmax(axis=-1)

Y_pred_flatten = Y_pred.flatten()
Y_pred_flatten = Y_pred_flatten[Y_test_flatten_tmp > 0]
Y_pred_flatten = [id2label[id] for id in Y_pred_flatten]


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
y_test = Y_test_flatten
y_pred = Y_pred_flatten


print(classification_report(y_test, y_pred))


print('Precision                                   : %.4f'%precision_score(y_test, y_pred, average='weighted'))
print('Recall                                      : %.4f'%recall_score(y_test, y_pred, average='weighted'))
print('F1-Score                                    : %.4f'%f1_score(y_test, y_pred, average='weighted'))
