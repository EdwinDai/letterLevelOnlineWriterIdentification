from models.model6 import NeuralNetwork
from data import Dataset_SVC2004
from torch.utils.data import DataLoader
from train import train, test
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import random
import numpy as np


logdir = r'./run/exp9'
writer = SummaryWriter(log_dir=logdir)

cuda = torch.device('cuda')


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
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss().cuda()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

epoch = 20

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