import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=8, hidden_size=64, num_layers=1,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1,
                             bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.W = nn.Parameter(torch.randn(64, 32))
        self.U = nn.Parameter(torch.randn(32, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv1d(300, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.AvgPool = nn.AvgPool1d(4)

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

    def forward_once(self, x):
        y1, _ = self.lstm1(x)
        y1 = self.dropout(y1)
        y1, _ = self.lstm2(y1)
        y1 = self.dropout(y1)

        y11 = self.softmax((self.tanh(y1 @ self.W)) @ self.U)
        y1 = torch.transpose(y1, -1, -2)
        y1 = y1 @ y11

        y2 = self.conv1(x)
        y22 = self.residual(y2)
        y2 = self.relu(y2 + y22)
        y2 = self.AvgPool(self.conv2(y2))
        y2 = y2.view(4, 64, 1)
        y = y1 + y2
        y = y.view(4, 1, 64)
        return y

    def forward(self, x):
        x1, x2 = x
        # x1 = x1.cuda()
        # x2 = x2.cuda()
        # x1 = x[:, 0]
        # x2 = x[:, 1]

        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y = torch.concat([y1, y2], dim=-1)
        y = self.DiscrimNet(y)
        return y
