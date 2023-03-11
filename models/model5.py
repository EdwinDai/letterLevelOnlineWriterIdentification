import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=32, num_layers=1,
                             bidirectional=False)  # [b,300,3] [b,300,30]
        self.conv1 = nn.Conv1d(300, 64, 3)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=64, num_layers=1,
                             bidirectional=False)  # [b,300,60] [b,300,120]

    def forward_once(self, x):
        y1, _ = self.lstm1(x)  # [b,300,30]
        y2 = self.conv1(x)
        y = y1 + y2
        return y

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()

        # x1 = x[:, 0]
        # x2 = x[:, 1]

        y1 = self.forward_once(x1)  # [b,300,180]
        y2 = self.forward_once(x2)  # [b,300,180]

        output = torch.concat([y1, y2], dim=-1)  # [b,300,180] [b,300,360]

        output, _ = self.lstm3(output)  # [b,300,180]
        output = self.together1(output)  # [b,300,120]
        output, _ = self.lstm4(output)  # [b,300,60]
        output = self.together2(output)  # [b,300,30]

        output = self.linear(output[:, -1, :])  # [b,1,30] [b,2]
        output = self.softmax(output)
        return output
