import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=16, num_layers=1, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)

        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(32, 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward_once(self, x):
        y, _ = self.lstm1(x)
        y = self.linear1(y[:, -1, :])
        y = self.dropout(y)
        y, _ = self.lstm2(y)
        return y

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        output = torch.concat([y1, y2], dim=1)
        output = self.linear2(output)
        output = self.softmax(output)
        return output
