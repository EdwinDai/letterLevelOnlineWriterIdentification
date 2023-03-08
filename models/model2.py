import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=30, num_layers=1,
                             bidirectional=False)  # [b,300,3] [b,300,30]
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=120, num_layers=1,
                             bidirectional=False)  # [b,300,60] [b,300,120]
        self.siamese1 = nn.Sequential(  # [b,300,30] [b,300,60]
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.siamese2 = nn.Sequential(  # [b,300,120] [b,300,180]
            nn.Linear(120, 180),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.lstm3 = nn.LSTM(input_size=360, hidden_size=180, num_layers=1,
                             bidirectional=False)  # [b,300,360] [b,300,180]
        self.lstm4 = nn.LSTM(input_size=120, hidden_size=60, num_layers=1,
                             bidirectional=False)  # [b,300,120] [b,300,60]
        self.together1 = nn.Sequential(  # [b,300,180] [b,300,120]
            nn.Linear(180, 120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.together2 = nn.Sequential(  # [b,300,60] [b,300,30]
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(p=0.1))

        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(30, 2)

    def forward_once(self, x):
        y, _ = self.lstm1(x)  # [b,300,30]
        y = self.siamese1(y)  # [b,300,60]
        y, _ = self.lstm2(y)  # [b,300,120]
        y = self.siamese2(y)  # [b,300,180]
        return y

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()

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
