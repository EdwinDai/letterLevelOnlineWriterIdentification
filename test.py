from models.model5 import NeuralNetwork
from data import Dataset_SVC2004_train,Dataset_SVC2004_test
from torch.utils.data import DataLoader
from train import train, test
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import random
import numpy as np
from utils import countDataDistribution


logdir = r'./run/exp13'
writer = SummaryWriter(log_dir=logdir)

cuda = torch.device('cuda')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 1

setup_seed(seed)

# dataset = Dataset_SVC2004()
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# train_dataset, test_dataset = random_split(
#     dataset=dataset,
#     lengths=[12162, 3040],
#     # lengths=[18880, 4720],
#     generator=torch.Generator().manual_seed(seed)
# )

train_dataset = Dataset_SVC2004_train()
test_dataset = Dataset_SVC2004_test()
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# model = NeuralNetwork()
model = NeuralNetwork().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# countDataDistribution(train_dataloader, test_dataloader)

epoch = 30
for i in range(epoch):
    train(train_dataloader, model, loss_fn, optimizer, writer=writer, currentEpoch=i)
    test(test_dataloader, model, loss_fn, currentEpoch=i, writer=writer)
    # scheduler.step()
#
# for x, y in train_dataloader:
#     x1, x2 = x
#     print(x1.shape)
#     print(x2.shape)
#     res = model(x)
#     print(res.shape)
#     # print(res)
#     print(y.shape)
#     print(y)
#     break
# paras = model.parameters()
# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))

# x = torch.randn(8, 2, 300, 3)
# writer.add_graph(model, input_to_model=x)
# writer.close()


