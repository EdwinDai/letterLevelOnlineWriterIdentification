import torch
from models.model2 import NeuralNetwork
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    x = torch.randn(8, 2, 300, 3)
    model = NeuralNetwork()
    writer = SummaryWriter('run/exp5')
    writer.add_graph(model, input_to_model=x)
    writer.close()
