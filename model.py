import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils
import numpy as np
USE_CUDA = False

class RNN(nn.Module):
    def __init__(self, num_items, hidden_size, n_layers):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_items = num_items

        self.embed = nn.Embedding(num_items, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, num_items)
        self.init_weights()

    def forward(self, x, hidden):

        seq_len = len(x)
        embedded = self.embed(x).view(seq_len, 1, -1)

        # Forward propagate RNN
        out, hidden = self.gru(embedded, hidden)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        out = self.fc(out)
        return out, hidden

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        # Init hidden states for rnn
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
