import torch.nn as nn
import torch

class Model_LSTM_POS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, inner_dropout, dropout, vectors):
        super(Model_LSTM_POS, self).__init__()
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(vectors))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional, dropout = inner_dropout )
        self.dropout = nn.Dropout(dropout)
        bidirectional = 2 if bidirectional else 1
        self.fc = nn.Linear(bidirectional * hidden_size , output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, (hidden, _) = self.lstm(x)
        x = self.fc(out)
        return nn.functional.log_softmax(x)