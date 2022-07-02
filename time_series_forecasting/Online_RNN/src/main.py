""" Main
    Zhymir Thompson 2022
    """
import os

import numpy as np
import torch
import pandas as pd

from forecaster.custom_models.forecaster_model import update_weights, run_forecaster, Forecaster
from forecaster.custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


# TODO separate discriminators or set entire program up to remove autograd in predictor
class TorchRNN(RecurrentNeuralNetworkTorch):
    def __init__(self, history_length, loss_fn, data_type=torch.float32):
        super(TorchRNN, self).__init__()
        self.dtype = data_type
        self.input_length = history_length
        self.rec_1 = torch.nn.RNN(history_length, 1, dtype=self.dtype)
        self.relu_1 = torch.nn.ReLU()
        self.loss_fn = loss_fn

    def forward(self, input_val):
        # input_val.dtype(self.dtype)
        return self.rec_1(input_val)

    def predict(self, input_val):
        return self.forward(input_val)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)


def load_data_numpy(filename: str):
    """ Read data with numpy.

        :param str filename: File path to data
        """
    return np.loadtxt(filename, skiprows=23)


def load_data_pandas(filename: str):
    return pd.read_table(filename, usecols=[0, 1, 2, 3], header=22)


def parse_data(filename: str):
    raise NotImplementedError


def copy_weights(model, model_2):
    """ """
    for (name_1, param_1), (name_2, param_2) in zip(model.named_parameters(), model_2.named_parameters()):
        if name_1 == name_2:  # this should always be true, faster to check all at once
            param_1.data = param_2.data


def main():
    history_length = 10  # how many points of data to use for inference
    # Load data
    filename = os.path.join(
        os.pardir, os.pardir, 'Dataset-4-Univariate-signal-with-non-stationarity',
        'data', 'data_III', 'Test 1.lvm')
    # Pick a method
    # data = parse_data(filename)
    data = load_data_numpy(filename)
    # Make models
    predictor = TorchRNN(history_length, loss_fn=None, data_type=torch.float32)
    learner = TorchRNN(history_length, loss_fn=torch.nn.MSELoss, data_type=torch.float32)
    update_weights(learner, predictor.named_parameters())
    # Train models
    training_data = torch.from_numpy(np.expand_dims(data[:100, 1], axis=0).astype(np.float32))
    run_forecaster(Forecaster(predictor, learner, torch.nn.MSELoss, None), training_data, history_length, 5)
    # Evaluate models


if __name__ == '__main__':
    main()
