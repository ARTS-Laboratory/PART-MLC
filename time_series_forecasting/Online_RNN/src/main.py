""" Main
    Zhymir Thompson 2022
    """

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from data_transform.signal_fft import sampling_frequency
from eval.metrics import trac, snr
from custom_models.forecaster_model import update_weights, run_forecaster, Forecaster
from custom_models.recurrent_nn_pytorch import RecurrentNeuralNetworkTorch
from model_selection.data_prep import get_training_data_sliding_window
from utils.toml_utils import load_toml


class TorchMLPRNN(RecurrentNeuralNetworkTorch):
    def __init__(self, history_length, loss_fn, num_layers=1, data_type=torch.float32):
        """

        :param int history_length: How many features the model observes at one time.
        :param loss_fn: Loss function to apply to model
        :type loss_fn:
        :param int num_layers: Number of layers model has. Defaults to 1.
        :param data_type: Type of data to feed into model
        :type data_type: torch.dtype
        """
        super(TorchMLPRNN, self).__init__()
        self.dtype = data_type
        self.input_length = history_length
        self.hidden_features = 64
        self.num_layers = num_layers
        self.hidden_1, self.hidden_2 = 64, 128
        self.hidden_3, self.hidden_4 = 128, 128
        self.hidden_5, self.hidden_last = 64, 16
        # self.hidden_7, self.hidden_8 = 2, 2
        # self.hidden_9, self.hidden_10 = 2, 2
        self.layer_1 = torch.nn.Linear(self.input_length, self.hidden_1, dtype=self.dtype)
        self.layer_2 = torch.nn.Linear(self.hidden_1, self.hidden_2, dtype=self.dtype)
        self.layer_3 = torch.nn.Linear(self.hidden_2, self.hidden_3, dtype=self.dtype)
        self.layer_4 = torch.nn.Linear(self.hidden_3, self.hidden_4, dtype=self.dtype)
        self.layer_5 = torch.nn.Linear(self.hidden_4, self.hidden_5, dtype=self.dtype)
        # self.layer_6 = torch.nn.Linear(self.hidden_5, self.hidden_6, dtype=self.dtype)
        self.layer_7 = torch.nn.Linear(self.hidden_5, self.hidden_last, dtype=self.dtype)
        self.rec_1 = torch.nn.RNN(
            self.hidden_last + self.input_length, self.hidden_features, num_layers=self.num_layers,
            dtype=self.dtype, batch_first=True, bias=False, nonlinearity='tanh')
        self.last_layer = torch.nn.Linear(self.hidden_features, 1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.loss_fn = loss_fn
        self.hidden_state = self.random_hidden()

    def forward(self, input_val, hidden):
        """ Return prediction from model.

            :param input_val: Input values to feed into model.
            :param hidden: Hidden state of the model.
            :type input_val: torch.Tensor. Should have size
             (sequence length, input size) or (batch, sequence length, input size)
            :returns: Tuple of prediction output and state.
            :rtype: tuple[torch.Tensor, torch.Tensor]

            """
        # input_val.dtype(self.dtype)
        val = self.relu(self.layer_1(input_val))
        val = self.relu(self.layer_2(val))
        val = self.relu(self.layer_3(val))
        val = self.relu(self.layer_4(val))
        val = self.relu(self.layer_5(val))
        val = self.relu(self.layer_7(val))
        out, hidden = self.rec_1(torch.cat((val, input_val), 1), hidden)
        out = self.tanh(self.last_layer(out))
        return out, hidden

    def predict(self, input_val, hidden):
        """ Return call to forward.

            Wrapper for forward method.
        """
        return self.forward(input_val, hidden)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)

    def random_hidden(self, batched=0, bidirect=False):
        if batched != 0:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                batched, self.hidden_features)
        else:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                self.hidden_features)

    def make_hidden_state(self, batched=0, bidirect=False):
        return self.random_hidden(batched, bidirect)


class TorchRNN(RecurrentNeuralNetworkTorch):
    def __init__(self, history_length, loss_fn, num_layers=1, data_type=torch.float32):
        """

        :param int history_length: How many features the model looks at at one time.
        :param loss_fn: Loss function to apply to model
        :type loss_fn:
        :param int num_layers: Number of layers model has. Defaults to 1.
        :param data_type: Type of data to feed into model
        :type data_type: torch.dtype
        """
        super(TorchRNN, self).__init__()
        self.dtype = data_type
        self.input_length = history_length
        self.hidden_features = 1
        self.num_layers = num_layers
        self.rec_1 = torch.nn.RNN(
            history_length, self.hidden_features, num_layers=self.num_layers,
            dtype=self.dtype, batch_first=True, bias=False, nonlinearity='tanh')
        self.loss_fn = loss_fn
        self.hidden_state = self.random_hidden()

    def forward(self, input_val, hidden):
        """ Return prediction from model.

            :param input_val: Input values to feed into model.
            :param hidden: Hidden state of the model.
            :type input_val: torch.Tensor. Should have size
             (sequence length, input size) or (batch, sequence length, input size)
            :returns: Tuple of prediction output and state.
            :rtype: tuple[torch.Tensor, torch.Tensor]

            """
        # input_val.dtype(self.dtype)
        out, hidden = self.rec_1(input_val, hidden)
        return out, hidden

    def predict(self, input_val, hidden):
        """ Return call to forward.

            Wrapper for forward method.
        """
        return self.forward(input_val, hidden)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)

    def random_hidden(self, batched=0, bidirect=False):
        if batched != 0:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                batched, self.hidden_features)
        else:
            return torch.randn(
                self.num_layers if not bidirect else 2 * self.num_layers,
                self.hidden_features)

    def make_hidden_state(self, batched=0, bidirect=False):
        return self.random_hidden(batched, bidirect)


class TorchMLP(RecurrentNeuralNetworkTorch):
    def __init__(self, input_size, loss_fn=None, data_type=torch.float32):
        super(TorchMLP, self).__init__()
        self.dtype = data_type
        self.loss_fn = loss_fn
        self.hidden_1, self.hidden_2 = 8, 8
        self.hidden_3, self.hidden_4 = 16, 16
        self.hidden_5, self.hidden_6 = 8, 8
        # self.hidden_7, self.hidden_8 = 2, 2
        # self.hidden_9, self.hidden_10 = 2, 2
        self.layer_1 = torch.nn.Linear(input_size, self.hidden_1, dtype=self.dtype)
        self.layer_2 = torch.nn.Linear(self.hidden_1, self.hidden_2, dtype=self.dtype)
        self.layer_3 = torch.nn.Linear(self.hidden_2, self.hidden_3, dtype=self.dtype)
        self.layer_4 = torch.nn.Linear(self.hidden_3, self.hidden_4, dtype=self.dtype)
        # self.layer_5 = torch.nn.Linear(self.hidden_4, self.hidden_5, dtype=self.dtype)
        # self.layer_6 = torch.nn.Linear(self.hidden_5, self.hidden_6, dtype=self.dtype)
        self.layer_7 = torch.nn.Linear(self.hidden_4, 1, dtype=self.dtype)

        self.relu = torch.nn.ReLU()

    # hidden_state is added as a parameter to remain unused
    def forward(self, input_val, hidden_state=None):
        val = self.layer_1(input_val)
        val = self.layer_2(self.relu(val))
        val = self.layer_3(self.relu(val))
        val = self.layer_4(self.relu(val))
        # val = self.layer_5(self.relu(val))
        # val = self.layer_6(self.relu(val))
        val = self.layer_7(self.relu(val))
        return torch.nn.Tanh()(val), None

    def predict(self, input_val, hidden_state):
        return self.forward(input_val, hidden_state)

    def loss(self, prediction, actual):
        return self.loss_fn(prediction, actual)

    def random_hidden(self, batched=0, bidirect=False):
        """ Generate random values for initial hidden state and initial cell state.

            :returns: Tuple of random values. Does not currently account for projections.
        """
        return None

    def make_hidden_state(self, batched=0, bidirect=False):
        return None


# TODO separate discriminators or set entire program up to remove autograd in predictor
# class TorchRNN(RecurrentNeuralNetworkTorch):
#     def __init__(self, history_length, loss_fn, data_type=torch.float32):
#         super(TorchRNN, self).__init__()
#         self.dtype = data_type
#         self.input_length = history_length
#         self.rec_1 = torch.nn.RNN(history_length, 1, dtype=self.dtype)
#         self.relu_1 = torch.nn.ReLU()
#         self.loss_fn = loss_fn
#
#     def forward(self, input_val):
#         # input_val.dtype(self.dtype)
#         return self.rec_1(input_val)
#
#     def predict(self, input_val):
#         return self.forward(input_val)
#
#     def loss(self, prediction, actual):
#         return self.loss_fn(prediction, actual)


def load_data_numpy(filename: str):
    """ Read data with numpy.

        :param str filename: File path to data
        :returns: numpy array at filename
        :rtype: np.ndarray
    """
    return np.loadtxt(filename, skiprows=23)


def load_data_pandas(filename: str):
    """ Load and return pandas DataFrame.

        :param str filename: Filename to load
        :returns: pandas DataFrame at filename
        :rtype: pd.DataFrame"""
    table = pd.read_table(filename, usecols=[0, 1, 2, 3], header=21, delim_whitespace=True)
    return table


def parse_data(filename: str):
    raise NotImplementedError


def copy_weights(model, model_2):
    """ """
    for (name_1, param_1), (name_2, param_2) in zip(model.named_parameters(), model_2.named_parameters()):
        if name_1 == name_2:  # this should always be true, faster to check all at once
            param_1.data = param_2.data


def plot_predict_v_actual(
        predicted, actual, param_time, save=False, show=True,
        result_dir=None):
    """ Plot predicted values vs actual and return figure."""
    res_dir = '' if result_dir is None else result_dir
    plt.figure()
    plot_time = param_time * 1_000
    plt.plot(plot_time, actual, label='Actual accel')
    plt.plot(plot_time, predicted, '--', label='Predicted accel')
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(res_dir, 'predict_v_actual.pdf'))
        plt.savefig(os.path.join(res_dir, 'predict_v_actual.png'), dpi=250)
    if show:
        plt.show()


def process_results(results):
    if isinstance(results, tuple):
        raise NotImplementedError
    else:
        results = np.asarray(list(map(lambda x: x.flatten().item(), results)))
        return results


def write_frame_to_latex(frame: pd.DataFrame, filename, folder=None):
    """ Writes reference frame and prediction frame to LaTeX files.

    Parameters
    ----------
    :param pd.DataFrame frame: Pandas Dataframe containing reference data metrics.
    :param filename: File name to write reference data LaTeX file.
    :type filename: str or None
    :param folder: Optional folder to prefix file names.
    :type folder: str or None


    """
    tmp_folder = folder if folder else ''
    frame.style.format(precision=3).hide(axis='index').to_latex(
        buf=os.path.join(tmp_folder, filename), hrules=True)


def save_outputs(
        predictions, actual, param_time, res_dir, metrics=None,
        metric_names=None, printed_metric_names=None):
    """ Save outputs of forecast to output files.

        :param np.ndarray predictions: Vector of predicted values.
        :param np.ndarray actual: Vector of measured values.
        :param np.ndarray param_time: Vector of time data for recorded values.
        :param res_dir: Directory path for results to be added.
        :param metrics: List of metric functions to apply to given data.
        :param metric_names: List of names of metrics given in metrics parameter.
        :param printed_metric_names: List of print friendly names for given metrics.
        :returns: None
    """
    # Save data
    np.savetxt(os.path.join(res_dir, 'predictions.txt'), predictions)
    np.save(os.path.join(res_dir, 'predictions.npy'), predictions)
    np.savetxt(
        os.path.join(res_dir, 'predictions.csv'), predictions, delimiter=',')
    np.save(os.path.join(res_dir, 'time.npy'), param_time)
    np.save(os.path.join(res_dir, 'actual.npy'), actual)
    # Metrics
    if metrics is not None and metric_names is not None:
        metrics_dict = {
            'metric': metric_names if printed_metric_names is None else printed_metric_names,
            'score': [metric(predictions, actual) for name, metric in zip(metric_names, metrics)]}
        # metrics_dict = {
        #     'metric': metric_names if printed_metric_names is None else printed_metric_names}
        # metrics_dict.update({
        #     'score': [metric(predictions, actual) for name, metric in zip(metric_names, metrics)]})
        metrics_frame = pd.DataFrame(metrics_dict, index=metric_names)
        write_frame_to_latex(
            metrics_frame, 'metrics.tex',
            folder=os.path.join(res_dir, 'latex_tables'))
        metrics_frame.to_csv(os.path.join(res_dir, 'metrics.csv'))


def get_args():
    """ Argument parsing"""
    parse = argparse.ArgumentParser()
    parse.add_argument('model', choices=['mlp', 'rnn', 'rnn_mlp', 'mlp_rnn'], default='rnn')
    parse.add_argument('config_file')
    # parse.add_argument('data_file', default=None)
    parse.add_argument('--make_paths', action='store_true')
    parse.add_argument('--results_dir', default=None)
    parse.add_argument('--save', action='store_true')
    args = parse.parse_args()
    res_dir = os.path.join(args.results_dir, args.model)
    if not os.path.exists(res_dir) and args.make_paths:
        os.makedirs(os.path.join(res_dir, 'latex_tables'))
        os.makedirs(os.path.join(res_dir, 'learner', 'latex_tables'))
    config = load_toml(args.config_file)
    training_config = config['training']
    history_length = training_config['window_size']
    time_skip = training_config['prediction_gap']
    time, train_x, train_y = prepare_data(
        training_config['filepath'],
        training_config['start_index'],
        training_config['prediction_gap'],
        training_config['length'] + (2 * history_length))
    if args.model == 'rnn':
        predictor = TorchRNN(
            history_length, loss_fn=None, num_layers=1,
            data_type=torch.float32)
        learner = TorchRNN(
            history_length, loss_fn=torch.nn.MSELoss(), num_layers=1,
            data_type=torch.float32)
    elif args.model == 'mlp':
        predictor = TorchMLP(
            history_length, loss_fn=None, data_type=torch.float32)
        learner = TorchMLP(
            history_length, loss_fn=torch.nn.MSELoss(), data_type=torch.float32)
    elif args.model == 'mlp_rnn':
        predictor = TorchMLPRNN(
            history_length, num_layers=4, loss_fn=None, data_type=torch.float32)
        learner = TorchMLPRNN(
            history_length, num_layers=4, loss_fn=torch.nn.MSELoss(), data_type=torch.float32)
    # Evaluate model
    results, results_2 = run_forecaster(
        Forecaster(predictor, learner, torch.nn.MSELoss(), None),
        train_x, history_length, time_skip)
    # Evaluate models
    data_length = len(results)
    data_length_2 = len(results_2)
    # print(data_length)
    results = np.asarray(list(map(lambda x: x.flatten().item(), results)))
    results_2 = np.asarray(list(map(lambda x: x.flatten().item(), results_2)))
    if args.save and res_dir is not None:
        save_outputs(
            results, train_y[-data_length:], time[-data_length:], res_dir,
            [lambda x, y: snr(x, y), trac], ['SNR$_dB$', 'TRAC'])
        plot_predict_v_actual(
            results, train_y[-data_length:], time[-data_length:], save=True,
            show=True, result_dir=res_dir)
        save_outputs(
            results_2, train_y[-data_length_2:], time[-data_length_2:], os.path.join(res_dir, 'learner'),
            [lambda x, y: snr(x, y), trac], ['SNR$_dB$', 'TRAC'])
        plot_predict_v_actual(
            results_2, train_y[-data_length_2:], time[-data_length_2:], save=True,
            show=True, result_dir=os.path.join(res_dir, 'learner'))


def prepare_data(data, start_idx, pred_gap, slice_length):
    """ Take data and return correct slices.

        :param data: Data to slice.
        :type data: str or pd.DataFrame
        :param int start_idx: Starting index for where to begin slice.
        :param int pred_gap: Number of points between last point used for
         inference and point to be predicted.
        :param int slice_length: Total length of data slice

    """
    # If with pandas
    if isinstance(data, str):  # If string assume it's a filename and load panda frame
        data = load_data_pandas(data)
    end = start_idx + pred_gap + slice_length + - 1
    time = data.loc[start_idx + pred_gap: end + pred_gap, 'X_Value'].array
    train_x = data.loc[start_idx: end, 'Acceleration'].array
    train_y = data.loc[
              start_idx + pred_gap: end + pred_gap, 'Acceleration'].array
    train_x = np.expand_dims(train_x, axis=0).astype(np.float32)
    # If with numpy
    # if isinstance(data, str):  # If string assume it's a filename and load panda frame
    #     data = pd.read_csv(data)
    # end = start_idx + pred_gap + slice_length + - 1
    # time = data[start_idx: end, 0]
    # train_x = data[start_idx: end, 3]
    # train_y = data[start_idx: end]

    return time, train_x, train_y


def main():
    # TODO train split function should be used to make life easy
    res_dir = os.path.join(os.pardir, 'results', 'main', '10_31_22', 'rnn')
    start_idx = 150_000
    history_length = 8  # how many points of data to use for inference
    time_skip = 10
    length = 20_000 + (2 * history_length)  # 200_014
    # Load data
    filename = os.path.join(
        os.pardir, 'data', 'Dataset-4-Univariate-signal-with-non-stationarity',
        'data', 'data_III', 'Test 1.lvm')
    # Pick a method
    data = load_data_pandas(filename)
    time = data.loc[start_idx: start_idx + time_skip + length - 1, 'X_Value'].array
    train_x = data.loc[start_idx: start_idx + length - 1 + time_skip, 'Acceleration'].array
    train_y = data.loc[start_idx: start_idx + length - 1 + time_skip, 'Acceleration'].array
    # train_x, train_y = get_training_data(data, start_idx, time_skip, length, 'Acceleration')
    # Make models
    predictor = TorchRNN(history_length, loss_fn=None, num_layers=1, data_type=torch.float32)
    learner = TorchRNN(history_length, loss_fn=torch.nn.MSELoss, num_layers=1, data_type=torch.float32)
    # predictor = TorchMLP(history_length, loss_fn=None, data_type=torch.float32)
    # learner = TorchMLP(history_length, loss_fn=torch.nn.MSELoss, data_type=torch.float32)

    update_weights(learner, predictor.named_parameters())
    # Train models
    new_training_data = np.expand_dims(train_x, axis=0).astype(np.float32)
    # training_data = torch.from_numpy(np.expand_dims(data.loc[:length, 'Acceleration'].array, axis=0).astype(np.float32))

    # training_data = torch.from_numpy(np.expand_dims(data[:10_000, 1], axis=0).astype(np.float32))
    results, results_2 = run_forecaster(
        Forecaster(predictor, learner, torch.nn.MSELoss, None),
        new_training_data, history_length, time_skip)
    # Evaluate models
    data_length = len(results)
    print(data_length)
    results = np.asarray(list(map(lambda x: x.flatten().item(), results)))
    save_outputs(
        results, train_y[-data_length:], time[-data_length:], res_dir,
        [lambda x, y: snr(x, y), trac], ['SNR$_dB$', 'TRAC'])
    plot_predict_v_actual(
        results, train_y[-data_length:], time[-data_length:], save=True,
        show=True, result_dir=res_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        get_args()
    else:
        main()
