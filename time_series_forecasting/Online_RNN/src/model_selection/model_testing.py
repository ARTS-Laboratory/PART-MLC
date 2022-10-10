""" Thompson 2022"""
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from eval.metrics import trac, snr
from main import load_data_pandas
from model_selection.model_architectures import TorchRNNExperiment, TorchLSTMExperiment
from model_selection.utils.data_prep import get_training_data
from utils.matplotlib_utils import save_fig
from custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch


def train_offline(model, optimizer, input_dataset, expected_set, epochs=1):
    """ Train model offline.

        :param model: Model to train.
        :param optimizer: Optimizer to update model.
        :type optimizer: torch.optim.Optimizer
        :param input_dataset:
        :type input_dataset: torch.Tensor
        :param expected_set:
        :type expected_set: torch.Tensor
        :param int epochs: Number of epochs to iterate over input_dataset.
         Defaults to 1.
    """
    losses = list()
    for epoch in range(epochs):
        hidden_state = model.random_hidden()
        for input_data, expected in zip(input_dataset, expected_set):
            optimizer.zero_grad()
            prediction, hidden_state = model.predict(input_data, hidden_state)
            loss = model.loss(prediction, expected)
            losses.append(loss)
            loss.backward()
            optimizer.step()
    return torch.as_tensor(losses)


def train_rnn_offline(model, optimizer, input_dataset, expected_set, sequence_length, epochs=1):
    """ Train RNN model offline and return loss.

        Model trains for given number of epochs. Each epoch, train on
        number of points equal to sequence_length
        :param model:
        :param optimizer:
        :param input_dataset:
        :param expected_set:
        :param sequence_length:
        :param epochs:
        :return: Tensor of loss values
        :rtype: torch.Tensor
    """
    losses = list()
    for epoch in range(epochs):
        for input_data, expected in zip(input_dataset, expected_set):
            optimizer.zero_grad()
            total_loss: torch.Tensor = 0.0
            hidden_state = model.random_hidden()
            for idx in range(sequence_length):
                prediction, hidden_state = model(input_data, hidden_state)
                total_loss += model.loss(prediction, expected)
            losses.append(total_loss)
            total_loss.backward()
            optimizer.step()
    return torch.as_tensor(losses)


def evaluate_rnn_model(model, input_dataset, expected_set, eval_fn):
    """ Evaluate model using given function.

        :param model: Model.
        :type model: RecurrentNeuralNetworkTorch
        :param input_dataset: Input data to feed to model. Should be 3 dimensional if batched, else 2
        :type input_dataset: torch.Tensor
        :param expected_set:
        :type expected_set: torch.Tensor
        :param eval_fn: Evaluation function.
        :type eval_fn: Callable
    """
    num_dims = len(input_dataset.size())
    if num_dims == 4:
        raise NotImplementedError('Written not tested.')
        predictions = list()
        for input_data in input_dataset:
            prediction, _ = model(input_data)
            predictions.append(prediction)
        predictions = torch.as_tensor(predictions)
    elif num_dims == 3:
        prediction, _ = model(input_dataset)
    if torch.is_same_size(predictions, expected_set):
        return eval_fn(
            expected_set.view(-1).detach().numpy(),
            predictions.view(-1).detach().numpy())
    else:
        raise ArithmeticError('Expected values do not have same size as predicted.')
    # predictions = model.predict(input_dataset)
    return eval_fn(expected_set, predictions)


def evaluate_rnn_model(model, input_dataset, expected_set, initial_hidden, eval_fn):
    """ Evaluate model using given function.

        :param model: Model.
        :type model: RecurrentNeuralNetworkTorch
        :param input_dataset: Input data to feed to model. Should be 3 dimensional if batched, else 2
        :type input_dataset: torch.Tensor
        :param expected_set:
        :type expected_set: torch.Tensor
        :param eval_fn: Evaluation function.
        :type eval_fn: Callable or List[Callable]
        :returns: Results of applying eval to predicted and expected sets
        :rtype: torch.Tensor or List[torch.Tensor]
    """
    num_dims = input_dataset.dim()
    if num_dims == 3:
        # raise NotImplementedError('Written not tested.')
        predictions = list()
        hidden = initial_hidden
        for input_data in input_dataset:
            prediction, hidden = model(input_data, hidden)
            predictions.append(prediction)
        predictions = torch.reshape(torch.as_tensor(predictions), expected_set.size())
    # elif num_dims == 3:
    #     predictions, _ = model.predict(input_dataset, initial_hidden)
    if torch.is_same_size(predictions, expected_set):
        if callable(eval_fn):
            return eval_fn(
                expected_set.view(-1).detach().numpy(),
                predictions.view(-1).detach().numpy())
        elif all(map(callable, eval_fn)):
            return [
                fn(expected_set.view(-1).detach().numpy(),
                   predictions.view(-1).detach().numpy()) for fn in eval_fn]
        else:
            raise NotImplementedError
    else:
        raise ArithmeticError('Expected values do not have same size as predicted.')


def loss_graph(losses):
    """ Return graph of loss over training.

        :param losses: Iterable of loss values.
        :type losses: list or np.ndarray
        :returns: Handle to figure
    """
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE)')  # This will need to change if loss isn't rmse
    return fig


def temp_plot_signal(time, signal):
    """ Plot and return handle to matplotlib plot of signal."""
    fig = plt.figure()
    plt.plot(time, signal)
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    return fig


def plot_predict_actual(time, predicted, actual):
    """ Plot and return figure comparing model prediction to actual signal.

        :param time: Time vector for x-axis.
        :type time: np.ndarray
        :param actual: Actual signal to plot.
        :type actual: np.ndarray
        :param predicted: Predicted signal to plot.
        :type predicted: np.ndarray
        :returns: Reference to plotted figure
        :rtype: plt.Figure
    """
    fig = plt.figure()
    plt.plot(time, actual, label='Actual signal')
    plt.plot(time, predicted, label='Prediction')
    plt.xlabel('Time (unit of seconds)')
    plt.ylabel('Acceleration (units)')
    plt.legend()
    return fig


def main():
    # Load data
    filename = os.path.join(
        os.pardir, os.pardir, 'data', 'Dataset-4-Univariate-signal-with-non-stationarity',
        'data', 'data_III', 'Test 1.lvm')
    data = load_data_pandas(filename)
    print(data.head())
    signal_plot = temp_plot_signal(data['X_Value'], data['Acceleration'])
    plt.show()
    # Data prep
    time = data.loc[:, 'X_Value'].array
    samp_freq = (time[-1] - time[0]) / time.size
    test_sequence_length = 1_000  # Total number of points to be trained/tested on
    prediction_gap = 100  # How far into the future model predicts
    train_index, test_index = 45, 15_000
    accel_str = 'Acceleration'
    test_time = data.loc[
                test_index + prediction_gap:
                test_index + prediction_gap + test_sequence_length - 1,
                'X_Value'].array
    train_data = torch.tensor(
        data.loc[
        train_index:train_index + test_sequence_length - 1, accel_str].array,
        dtype=torch.float32).view(-1, 1, 1)
    train_actual = torch.tensor(
        data.loc[
        train_index + prediction_gap:
        train_index + prediction_gap + test_sequence_length - 1,
        accel_str].array, dtype=torch.float32).view(-1, 1, 1)
    test_data = torch.tensor(
        data.loc[
        test_index:test_index + test_sequence_length - 1, accel_str].array,
        dtype=torch.float32).view(-1, 1, 1)
    test_actual = torch.tensor(
        data.loc[
        test_index + prediction_gap:
        test_index + prediction_gap + test_sequence_length - 1,
        accel_str].array, dtype=torch.float32).view(-1, 1, 1)
    # Model construction and training
    rnn_model = TorchRNNExperiment(history_length=1, loss_fn=torch.nn.MSELoss(), num_layers=2, data_type=torch.float32)
    optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-4)

    rnn_batches = test_data.size(dim=0)
    rnn_trac = evaluate_rnn_model(rnn_model, test_data, test_actual, rnn_model.random_hidden(rnn_batches), eval_fn=trac)
    rnn_snr = evaluate_rnn_model(rnn_model, test_data, test_actual, rnn_model.random_hidden(rnn_batches), eval_fn=lambda x, y: snr(x, y, samp_freq))
    print(f'RNN - TRAC score: {rnn_trac}, SNR score: {rnn_snr}')
    print('Starting training')
    rnn_loss = train_rnn_offline(
        rnn_model, optimizer, train_data, expected_set=train_actual,
        sequence_length=50, epochs=8)  # This return may change to dict
    print('End of training')
    rnn_loss = np.sqrt(torch.tensor(rnn_loss).detach().numpy())
    rnn_loss_fig = loss_graph(rnn_loss)
    test_predictions = rnn_model.forward(
        test_data, rnn_model.random_hidden(batched=test_data.size(dim=0)))[0].view(-1).detach().numpy()
    rnn_compare_fig = plot_predict_actual(test_time, test_predictions, test_actual.view(-1).detach().numpy())
    # plt.savefig('')
    rnn_trac = evaluate_rnn_model(rnn_model, test_data, test_actual, rnn_model.random_hidden(rnn_batches), eval_fn=trac)
    rnn_snr = evaluate_rnn_model(rnn_model, test_data, test_actual, rnn_model.random_hidden(rnn_batches), eval_fn=lambda x, y: snr(x, y, samp_freq))
    print(f'RNN - TRAC score: {rnn_trac}, SNR score: {rnn_snr}')
    # plt.show()
    lstm_model = TorchLSTMExperiment(history_length=1, loss_fn=torch.nn.MSELoss(), data_type=torch.float32)
    lstm_optim = torch.optim.SGD(lstm_model.parameters(), lr=1e-4)
    print('Starting training')
    lstm_losses = train_rnn_offline(
        lstm_model, lstm_optim, train_data, expected_set=train_actual,
        sequence_length=50, epochs=8)  # This return may change to dict
    print('End of training')
    lstm_losses = np.sqrt(torch.tensor(lstm_losses).detach().numpy())
    lstm_loss_fig = loss_graph(lstm_losses)
    test_predictions = lstm_model.forward(
        test_data, lstm_model.random_hidden(batched=test_data.size(dim=0)))[0].view(-1).detach().numpy()
    lstm_compare_fig = plot_predict_actual(test_time, test_predictions, test_actual.view(-1).detach().numpy())

    rnn_trac = evaluate_rnn_model(lstm_model, test_data, test_actual, lstm_model.random_hidden(rnn_batches), eval_fn=trac)
    rnn_snr = evaluate_rnn_model(lstm_model, test_data, test_actual, lstm_model.random_hidden(rnn_batches), eval_fn=lambda x, y: snr(x, y, samp_freq))
    print(f'LSTM - TRAC score: {rnn_trac}, SNR score: {rnn_snr}')
    plt.show()


if __name__ == '__main__':
    main()
