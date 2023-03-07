import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.siamese = nn.Sequential(
            nn.LSTM(input_size=3, hidden_size=32, num_layers=2, bidirectional=False),  # [b,300,3] [b,300,32]
            nn.Linear(32, 64),  # [b,300,32] [b,300,64]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=False),  # [b,300,64] [b,300,128]
            nn.Linear(128, 256),  # [b,300,128] [b,300,256]
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.together = nn.Sequential(  # [b,300,512] [b,300,32]
            nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=False),  # [b,300,512] [b,300,256]
            nn.Linear(256, 128),  # [b,300,256] [b,300,128]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LSTM(input_size=128, hidden_size=64, num_layers=2, bidirectional=False),  # [b,300,128] [b,300,64]
            nn.Linear(64, 32),  # [b,300,64] [b,300,32]
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(32, 2)

    def forward_once(self, x):
        y, _ = self.siamese(x)  # [b,300,3] [b,300,256]
        return y

    def forward(self, x):
        x1, x2 = x
        x1 = x1.cuda()
        x2 = x2.cuda()
        y1 = self.forward_once(x1)  # [b,300,256]
        y2 = self.forward_once(x2)  # [b,300,256]
        output = torch.concat([y1, y2], dim=-1)  # [b,300,256] [b,300,512]
        output, _ = self.together(output)  # [b,300,32]
        output = self.linear2(output[:, -1, :])  # [b, 1,32] [b,2]
        output = self.softmax(output)
        return output
