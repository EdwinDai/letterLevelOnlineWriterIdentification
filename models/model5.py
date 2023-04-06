import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=32, num_layers=2,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=48, num_layers=2,
                             bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)

        self.avePool = nn.AvgPool1d(2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv1d(300, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()

        )
        self.residual = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.DiscrimNet = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, padding=1),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.Linear(64, 2),
            nn.Softmax(-1)
        )
        self.linear1 = nn.Linear(192, 48)
        self.linear2 = nn.Linear(48, 2)

    def forward_once(self, x):
        y1, _ = self.lstm1(x)
        y1 = self.dropout(y1)
        y1, _ = self.lstm2(y1)
        y1 = self.dropout(y1)
        y1 = y1[:, -1, :]  # [-1,1,96]

        y2 = self.conv1(x)  # [-1,32,3]
        y21 = y2.view(-1, 1, 96)

        y22 = self.residual(y2)  # [-1,32,3]
        y22 = y22.view(-1, 1, 96)

        y2 = self.relu(y21 + y22)
        # y2 = self.avePool(y2)  # [-1,1,96]
        y2 = y2.view(-1, 96)

        y = y1 + y2
        return y

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
