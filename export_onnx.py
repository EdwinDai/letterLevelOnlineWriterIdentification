import torch
from model import NeuralNetwork


if __name__ == '__main__':
    x1 = torch.randn(300, 3)
    x2 = torch.randn(300, 3)
    input1 = (x1, x2)
    model = NeuralNetwork()
    torch.onnx.export(model, input1, 'model.onnx')
