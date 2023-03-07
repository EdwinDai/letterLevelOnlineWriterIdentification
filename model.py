import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=80, num_layers=4, bidirectional=False)

        self.linear1 = nn.Linear(80, 2)
        self.softmax = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(4,2)


    def forward_once(self, x):
        y, _ = self.lstm(x)
        y = self.linear1(y[:, -1, :])
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
