import torch
import onnx
from model import NeuralNetwork


def export_model_from_pytorch_to_onnx(pytorch_model, onnx_model_name):
    batch_size = 1
    # input to the model
    x1 = torch.randn(8, 300, 3)
    x2 = torch.randn(8, 300, 3)
    x = (x1, x2)
    out = pytorch_model(x)
    # print("out:", out)

    # export the model
    torch.onnx.export(pytorch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_model_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=9,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def verify_onnx_model(onnx_model_name):
    # model is an in-memory ModelProto
    model = onnx.load(onnx_model_name)
    # print("the model is:\n{}".format(model))

    # check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("  the model is invalid: %s" % e)
        exit(1)
    else:
        print("  the model is valid")


if __name__ == '__main__':
    model = NeuralNetwork()
    export_model_from_pytorch_to_onnx(model, 'siameseLSTM4')
