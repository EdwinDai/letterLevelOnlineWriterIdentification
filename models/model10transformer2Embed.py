import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=48, hidden_size=32, num_layers=2,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=48, num_layers=2,
                             bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(48, 4, 32), num_layers=2)

        self.avePool = nn.AvgPool1d(2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.linear3 = nn.Embedding(3, 48)

        self.linear1 = nn.Linear(192, 96)
        self.linear2 = nn.Linear(96, 2)

    def forward_once(self, x):
        y2 = self.linear3(x)  # [-1,300,48]
        # y2 = y2.permute(1, 0, 2)
        y2 = self.transformer(y2)

        y1, _ = self.lstm1(y2)
        y1 = self.dropout(y1)
        y1, _ = self.lstm2(y1)
        y1 = self.dropout(y1)
        y1 = y1[:, -1, :]  # [-1,1,96]

        return y1

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()
        # x1 = x[:, 0]
        # x2 = x[:, 1]

        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y = torch.concat([y1, y2], dim=-1)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.softmax(y)
        return y
