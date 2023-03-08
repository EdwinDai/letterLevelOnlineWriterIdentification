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
                             bidirectional=False)  # [b,300,3] [b,300,32]
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
                             bidirectional=False)  # [b,300,64] [b,300,128]
        self.siamese1 = nn.Sequential(  # [b,300,32] [b,300,64]
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.siamese2 = nn.Sequential(  # [b,300,128] [b,300,256]
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.lstm3 = nn.LSTM(input_size=512, hidden_size=256, num_layers=2,
                             bidirectional=False)  # [b,300,512] [b,300,256]
        self.lstm4 = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,
                             bidirectional=False)  # [b,300,128] [b,300,64]
        self.together1 = nn.Sequential(  # [b,300,256] [b,300,128]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.together2 = nn.Sequential(  # [b,300,64] [b,300,32]
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1))

        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(32, 2)

    def forward_once(self, x):
        y, _ = self.lstm1(x)  # [b,300,32]
        y = self.siamese1(y)  # [b,300,64]
        y, _ = self.lstm2(y)  # [b,300,128]
        y = self.siamese2(y)  # [b,300,256]
        return y

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()

        y1 = self.forward_once(x1)  # [b,300,256]
        y2 = self.forward_once(x2)  # [b,300,256]

        output = torch.concat([y1, y2], dim=-1)  # [b,300,256] [b,300,512]

        output, _ = self.lstm3(output)  # [b,300,256]
        output = self.together1(output)  # [b,300,128]
        output, _ = self.lstm4(output)  # [b,300,64]
        output = self.together2(output)  # [b,300,32]

        output = self.linear(output[:, -1, :])  # [b,1,32] [b,2]
        output = self.softmax(output)
        return output
