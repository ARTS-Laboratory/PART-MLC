""" Recurrent Neural Network Module
    Zhymir Thompson 2022
    """
import numpy


class RecurrentNeuralNetwork:
    def __init__(self):
        raise NotImplementedError

    def predict(self, input_val):
        raise NotImplementedError

    def loss(self, prediction, actual):
        raise NotImplementedError


def compute_loss(prediction, actual, loss_fn):
    """ Return loss between prediction and truth given loss function.
        :param prediction: array
        :param actual: array
        :param loss_fn: function
        :return output: array
        """
    return loss_fn(prediction, actual)


def calculate_gradients():
    """ Return gradients for parameters based on some loss."""
    raise NotImplementedError


def new_parameters():
    """ Return parameters updated by some set of gradients and old parameters."""
    raise NotImplementedError


if __name__ == '__main__':
    pass
