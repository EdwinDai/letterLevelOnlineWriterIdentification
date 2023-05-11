from models.model10_2dre import NeuralNetwork
from data2d import Dataset_SVC2004_train, Dataset_SVC2004_test
from torch.utils.data import DataLoader
from train2d import train, test
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import random
import numpy as np
from utils import countDataDistribution
from My_loss import My_loss

logdir = r'./run/exp17'
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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# model = NeuralNetwork()
# loss_fn = My_loss()

model = NeuralNetwork().cuda()
loss_fn = My_loss().cuda()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# countDataDistribution(train_dataloader, test_dataloader)

epoch = 50
for i in range(epoch):
    train(train_dataloader, model, loss_fn, optimizer, writer=writer, currentEpoch=i)
    test(test_dataloader, model, loss_fn, currentEpoch=i, writer=writer)
    # scheduler.step()

# for x, y in train_dataloader:
#     x1, x2, x3 = x
#     res = model(x)
#     print(res.shape)

# print(res)
# print(y.shape)
# print(y)
#     break
# paras = model.parameters()
# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))
# for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name + " is trainable")
#         else:
#             print(name + " is not trainable")
# x = torch.randn(8, 2, 300, 3)
# writer.add_graph(model, input_to_model=x)
# writer.close()

# epoch = 1
# for i in range(epoch):
#     train(train_dataloader, model, loss_fn, optimizer, currentEpoch=i)
#     test(test_dataloader, model, loss_fn, currentEpoch=i)