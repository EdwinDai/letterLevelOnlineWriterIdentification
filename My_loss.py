import torch
import torch.nn as nn


class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, anchor, pos, test, y):
        disa = self.pdist(anchor, pos)
        dist = self.pdist(anchor, test)
        minus = dist - disa
        mask = minus <= 0
        # pred = 1 if (disa - dist) <= 1 else 0
        pred = torch.where(mask, torch.ones_like(minus), -torch.ones_like(minus))
        # res = torch.mean(torch.relu(torch.matmul(-y.to(torch.float32).t(), (disa - dist)) + 1))
        res = torch.mean(torch.relu(disa - dist + 1))
        return res, pred, disa, dist, minus, y


if __name__ == '__main__':
    pdist = nn.PairwiseDistance(p=2)
    input1 = torch.randn(2, 4)
    print(input1)
    input2 = torch.randn(2, 4)
    print(input2)
    input3 = torch.randn(2, 4)
    print(input3)
    # print('input1 - input2', input1 - input2)
    # pow = torch.pow((input1 - input2), 2)
    # print('pow', pow)
    # disa = torch.sqrt(torch.sum(pow, dim=1))
    # print('disa', disa)
    # output = pdist(input1, input2)
    # print('output', output)
    loss = My_loss()
    loss(input1, input2, input3, -1)
