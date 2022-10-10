""" Recurrent Neural Network in PyTorch
    Zhymir Thompson 2022

    An abstract class for the recurrent Neural Network in PyTorch.
    Allows compatibility with other parts of the codebase.
    """
import torch


class RecurrentNeuralNetworkTorch(torch.nn.Module):
    def __init__(self):
        super(RecurrentNeuralNetworkTorch, self).__init__()

    def predict(self, input_val):
        raise NotImplementedError

    def loss(self, prediction, actual):
        raise NotImplementedError

    def make_hidden_state(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
