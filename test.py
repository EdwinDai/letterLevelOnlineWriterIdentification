from model import NeuralNetwork
from data import Dataset_SVC2004
from torch.utils.data import DataLoader
from train import train, test
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import random
import os
import numpy as np

logdir = r'./run/exp1'
writer = SummaryWriter(log_dir=logdir)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

dataset = Dataset_SVC2004()
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[25600, 6400],
    generator=torch.Generator().manual_seed(1)
)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = NeuralNetwork().cuda()

loss_fn = nn.CrossEntropyLoss().cuda()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters())
epoch = 1
for i in range(epoch):
    train(train_dataloader, model, loss_fn, optimizer, writer=writer, currentEpoch=i)
    test(test_dataloader, model, loss_fn, currentEpoch=i, writer=writer)

# for x, y in dataloader:
#     # print(x.shape)
#     res = model(x)
#     print(res.shape)
#     print(res)
#     print(y)
#     break