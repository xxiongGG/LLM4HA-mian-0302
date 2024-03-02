import torch
from torch import nn
from torch.autograd import Variable


class RNN_Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(RNN_Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # print(h0.shape)
        out, _ = self.rnn(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res


class LSTM_Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(LSTM_Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.lstm = nn.LSTM(input_size=num_questions * 2, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        out, _ = self.lstm(x)
        res = torch.sigmoid(self.fc(out))
        return res


class GRU_Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(GRU_Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.gru = nn.GRU(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # print(h0.shape)
        out, _ = self.gru(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res
