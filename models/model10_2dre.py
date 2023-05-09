import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))  # [-1, 499]
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 3)
        self.fc3 = nn.Linear(111, 1024)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(84, 42, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv3 = nn.Conv2d(102, 51, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        # dense1
        self.dense1Conv11 = nn.Conv2d(24, 24, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense1Conv12 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense11 = nn.Sequential(
            self.dense1Conv11,
            self.dense1Conv12
        )

        self.dense1Conv21 = nn.Conv2d(24, 24, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense1Conv22 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense12 = nn.Sequential(
            self.dense1Conv21,
            self.dense1Conv22
        )

        self.dense1Conv31 = nn.Conv2d(24, 24, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense1Conv32 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense13 = nn.Sequential(
            self.dense1Conv31,
            self.dense1Conv32
        )

        self.dense1Conv41 = nn.Conv2d(24, 24, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense1Conv42 = nn.Conv2d(24, 84, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense14 = nn.Sequential(
            self.dense1Conv41,
            self.dense1Conv42
        )

        # dense2
        self.dense2Conv11 = nn.Conv2d(42, 42, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense2Conv12 = nn.Conv2d(42, 42, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense21 = nn.Sequential(
            self.dense2Conv11,
            self.dense2Conv12
        )

        self.dense2Conv21 = nn.Conv2d(42, 42, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense2Conv22 = nn.Conv2d(42, 42, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense22 = nn.Sequential(
            self.dense2Conv21,
            self.dense2Conv22
        )

        self.dense2Conv31 = nn.Conv2d(42, 42, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense2Conv32 = nn.Conv2d(42, 42, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense23 = nn.Sequential(
            self.dense2Conv31,
            self.dense2Conv32
        )

        self.dense2Conv41 = nn.Conv2d(42, 42, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense2Conv42 = nn.Conv2d(42, 102, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense24 = nn.Sequential(
            self.dense2Conv41,
            self.dense2Conv42
        )

        # dense3
        self.dense3Conv11 = nn.Conv2d(51, 51, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense3Conv12 = nn.Conv2d(51, 51, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense31 = nn.Sequential(
            self.dense3Conv11,
            self.dense3Conv12
        )

        self.dense3Conv21 = nn.Conv2d(51, 51, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense3Conv22 = nn.Conv2d(51, 51, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense32 = nn.Sequential(
            self.dense3Conv21,
            self.dense3Conv22
        )

        self.dense3Conv31 = nn.Conv2d(51, 51, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense3Conv32 = nn.Conv2d(51, 51, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense33 = nn.Sequential(
            self.dense3Conv31,
            self.dense3Conv32
        )

        self.dense3Conv41 = nn.Conv2d(51, 51, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.dense3Conv42 = nn.Conv2d(51, 111, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.dense34 = nn.Sequential(
            self.dense3Conv41,
            self.dense3Conv42
        )

    def forward_once(self, x):
        y = self.gap1(x).squeeze()  # [-1, 3, 1, 1]
        y = self.fc1(y)
        y = self.sigmoid(self.relu(y))
        y = self.fc2(y)
        y = self.sigmoid(self.relu(y)).view(-1,3,1,1)
        cwl = x * y

        cnn1 = self.conv1(cwl)
        step1 = self.dense11(cnn1)
        step2 = self.dense12(step1)
        step3 = self.dense13(step1 + step2)
        step4 = self.dense14(step1 + step2 + step3)

        cnn2 = self.conv2(step4)
        cnn2 = self.pool1(cnn2)
        step1 = self.dense21(cnn2)
        step2 = self.dense22(step1)
        step3 = self.dense23(step1 + step2)
        step4 = self.dense24(step1 + step2 + step3)

        cnn3 = self.conv3(step4)
        cnn3 = self.pool2(cnn3)
        step1 = self.dense31(cnn3)
        step2 = self.dense32(step1)
        step3 = self.dense33(step1 + step2)
        step4 = self.dense34(step1 + step2 + step3)

        res = self.gap2(step4).squeeze()
        res = self.fc3(res)
        return res

    def forward(self, x):
        anchor, pos, test = x
        anchor = anchor.cuda()
        pos = pos.cuda()
        test = test.cuda()

        anchor = anchor.permute(0, 3, 1, 2).to(torch.float32)
        pos = pos.permute(0, 3, 1, 2).to(torch.float32)
        test = test.permute(0, 3, 1, 2).to(torch.float32)

        anchorres = self.forward_once(anchor)
        posres = self.forward_once(pos)
        testres = self.forward_once(test)
        return anchorres, posres, testres
