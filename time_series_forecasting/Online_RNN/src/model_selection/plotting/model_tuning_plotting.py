""" Thompson 2022

    Module to hold different types of plot for tuned models.
"""
import argparse
import os
from typing import List

import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

from main import load_data_pandas
from model_selection.model_architectures import TorchRNNExperiment, TorchLSTMExperiment
from model_selection.model_testing import train_rnn_offline
from model_selection.model_tuning_constants import CONFIG_DATA_PATH, CONFIG_GAP, CONFIG_LENGTH, ACCEL_INDEX, \
    CONFIG_START, CONFIG_TRAINING
from model_selection.utils.data_prep import get_training_data
from utils.matplotlib_utils import save_fig
from utils.toml_utils import load_toml


def plot_num_layers_vs_sequence_length_rnn(folder=None, save=False, show=False):
    """ Return 3D plot of model accuracy over varying number of model layers and sequence length.

        :returns: Figure to 3D plot.
    """
    config = load_toml(os.path.join(os.curdir, 'configs/rnn_config.toml'))[CONFIG_TRAINING]
    data = load_data_pandas(config[CONFIG_DATA_PATH])
    train_data, train_actual = get_training_data(
        data, config[CONFIG_START], config[CONFIG_GAP],
        config[CONFIG_LENGTH], ACCEL_INDEX)

    num_layers_range = [x for x in range(1, 11)]
    sequence_length_range = [x for x in range(1, 21)]

    xv, yv = np.meshgrid(num_layers_range, sequence_length_range)
    z = list()
    for x_range, y_range in zip(xv, yv):
        for x, y in zip(x_range, y_range):
            rnn_model = TorchRNNExperiment(history_length=1, loss_fn=torch.nn.MSELoss(), num_layers=x)
            loss = train_model(rnn_model, y, train_data, train_actual)
            z.append(loss)
            del rnn_model
    z = np.array(z).reshape(xv.shape)
    figure = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xv, yv, z, cstride=1, rstride=1, cmap='viridis', edgecolor='none')
    # ax.contour3D(xv, yv, z, 100)
    ax.set_xlabel('Number of model layers')
    ax.set_ylabel('Length of observed sequence')
    ax.set_zlabel('Loss (MSE)')
    if save:
        save_fig(figure, 'n_layers_v_seq_length.pdf', folder=folder)
        save_fig(figure, 'n_layers_v_seq_length.pdf', folder=folder, save_kwargs={'dpi': 200})
    if show:
        plt.show()
    return figure


def plot_predict_gap_vs_sequence_length_rnn(folder=None, save=False, show=False):
    """ Return 3D plot of model accuracy over varying number of model layers and sequence length.

        :returns: Figure to 3D plot.
    """
    config = load_toml(os.path.join(os.curdir, 'configs/rnn_config.toml'))[CONFIG_TRAINING]
    data = load_data_pandas(config[CONFIG_DATA_PATH])

    prediction_gap_range = [x for x in range(1, 101)]
    sequence_length_range = [x for x in range(1, 21)]

    xv, yv = np.meshgrid(prediction_gap_range, sequence_length_range)
    z = list()
    for x_range, y_range in zip(xv, yv):
        for x, y in zip(x_range, y_range):
            rnn_model = TorchRNNExperiment(history_length=1, loss_fn=torch.nn.MSELoss(), num_layers=1)
            train_data, train_actual = get_training_data(
                data, config[CONFIG_START], int(x),
                config[CONFIG_LENGTH], ACCEL_INDEX)
            loss = train_model(rnn_model, y, train_data, train_actual)
            del rnn_model
            z.append(loss)
    z = np.array(z).reshape(xv.shape)
    figure = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xv, yv, z, cstride=1, rstride=1, cmap='viridis', edgecolor='none')
    # ax.contour3D(xv, yv, z, 100)
    ax.set_xlabel('Prediction gap')
    ax.set_ylabel('Length of observed sequence')
    ax.set_zlabel('Loss (MSE)')
    if save:
        save_fig(figure, 'prediction_gap_v_seq_length.pdf', folder=folder)
        save_fig(figure, 'prediction_gap_v_seq_length.pdf', folder=folder, save_kwargs={'dpi': 200})
    if show:
        plt.show()
    return figure


def train_model(model, sequence_length, train_data, train_actual):
    """

    :param model:
    :param sequence_length:
    :type sequence_length: int
    :param torch.Tensor train_data:
    :param torch.Tensor train_actual:
    """
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # Training
    epochs = 1
    rnn_loss: List[torch.Tensor] = train_rnn_offline(
        model, optimizer, train_data, expected_set=train_actual,
        sequence_length=sequence_length, epochs=epochs)
    return torch.sum(rnn_loss.flatten()).item()


# def parse():
#     """ Parser for command line."""
#     parser = argparse.ArgumentParser('Program to make different hyperparameter interpolation plots.')
#     parser.add_argument('plot', type=)
#
#     parser.parse_args()

def main():
    print('first plot')
    plot_num_layers_vs_sequence_length_rnn()
    plt.savefig('./results/7_25_22/n_layers_v_seq_length.pdf')
    plt.savefig('./results/7_25_22/n_layers_v_seq_length.png', dpi=300)
    # plt.show()
    print('second plot')
    plot_predict_gap_vs_sequence_length_rnn()
    plt.savefig('./results/7_25_22/prediction_gap_v_seq_length.pdf')
    plt.savefig('./results/7_25_22/prediction_gap_v_seq_length.png', dpi=300)
    print('displaying')
    plt.show()


if __name__ == '__main__':
    main()
