""" Thompson 2022"""
import os
import argparse
import sys

import optuna
import torch

from optuna.trial import Trial as TrialType
from typing import List

from main import load_data_pandas
from model_selection.model_architectures import TorchRNNExperiment, TorchLSTMExperiment
from model_selection.model_testing import train_rnn_offline
from model_selection.model_tuning_constants import OBJECTIVE_CHOICES, CONFIG_TRAINING, CONFIG_DATA_PATH, CONFIG_START, \
    ACCEL_INDEX, CONFIG_LENGTH, CONFIG_GAP
from model_selection.utils.data_prep import get_training_data
from utils.json_utils import save_json_file
from utils.toml_utils import load_toml


def rnn_objective(trial: TrialType, config):
    # Define model
    num_layers = trial.suggest_int('num_layers', 1, 10)
    model = TorchRNNExperiment(1, torch.nn.MSELoss(), num_layers=num_layers, data_type=torch.float32)
    # Begin model training
    score = training(trial, model, config)
    return score


def lstm_objective(trial: TrialType, config):
    # Define model
    num_layers = trial.suggest_int('num_layers', 1, 10)
    model = TorchLSTMExperiment(1, torch.nn.MSELoss(), num_layers=num_layers, data_type=torch.float32)
    # Begin model training
    score = training(trial, model, config)
    return score


def gru_objective(trial: TrialType, config):
    raise NotImplementedError
    # Begin model training
    training()


# def get_data(filenames, folder=None):
#     """ """
#     # Load data
#     if isinstance(filenames, str):
#         # filename = os.path.join(
#         #     os.pardir, os.pardir, 'data', 'Dataset-4-Univariate-signal-with-non-stationarity',
#         #     'data', 'data_III', 'Test 1.lvm')
#         data = load_data_pandas(filenames)
#     else:
#         raise NotImplementedError('Multiple file loading not implemented yet.')
#     return data


def training(trial: TrialType, model, config):
    """

    :param trial:
    :type trial: TrialType
    :param model:
    :param config:
    """
    # TODO Fail early if config doesn't exist maybe
    # Optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    # optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-8, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    # Training
    sequence_length = trial.suggest_int('sequence_length', 1, 20, step=5)
    epochs = 1
    data = load_data_pandas(config[CONFIG_DATA_PATH])
    train_data, train_actual = get_training_data(
        data, config[CONFIG_START], config[CONFIG_GAP],
        config[CONFIG_LENGTH], ACCEL_INDEX)
    rnn_loss: List[torch.Tensor] = train_rnn_offline(
        model, optimizer, train_data, expected_set=train_actual,
        sequence_length=sequence_length, epochs=epochs)
    return torch.mean(rnn_loss.flatten()).item()


def run_study(objective, n_trials=100, show_progress_bar=False):
    """
    :param objective:
    :param n_trials:
    :param show_progress_bar:
    :return:
    """
    # new_kw = {'n_trials': 100, 'timeout': None, 'n_jobs': 1, 'show_progress_bar': False}
    # new_kw.update(kwargs)
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    print(f'Best parameters: {study.best_params}')
    return study


def write_study(study: optuna.Study, filename):
    """ Write study data to output file"""
    save_json_file(study.best_params, filename)


def parse():
    """ Parse arguments and pass to choose_model_tuning."""
    # Which objective function to use
    # Where to get data from
    # Number of trials
    # Progress flag
    # where to write results
    parser = argparse.ArgumentParser(prog='Model Tuner', description='Tuning CLI')
    parser.add_argument('objective', choices=OBJECTIVE_CHOICES)
    parser.add_argument('config_path', type=str)
    parser.add_argument('output', default='output.json', type=str)
    parser.add_argument('--n_trials', default=100, type=int)
    parser.add_argument('--show_progress', action='store_true')
    args = parser.parse_args()
    config = load_toml(args.config_path)
    train_config = config[CONFIG_TRAINING]
    study = choose_model_tuning(args.objective, train_config, args.n_trials, args.show_progress)
    write_study(study, args.output)
    return args


def choose_model_tuning(objective_string: str, config, n_trials=100, show_progress_bar=False):
    """ Call :func:`run_study` with objective function chosen by objective_string.

        :param str objective_string: Which objective function to use.
        :param config:
        :param int n_trials:
        :param bool show_progress_bar:
        Options can be found in model_tuning_constants as constant
         :const:`OBJECTIVE_CHOICES`.
    """
    if objective_string == 'rnn':
        fn = rnn_objective
    elif objective_string == 'lstm':
        fn = lstm_objective
    elif objective_string == 'gru':
        raise NotImplementedError('GRU has not been implemented yet.')
    else:
        raise NotImplementedError(f'"{objective_string}" is not currently an option')
    study = run_study(
        lambda x: fn(x, config), n_trials=n_trials,
        show_progress_bar=show_progress_bar)
    return study


def main():
    print(os.getcwd())
    num_args = len(sys.argv)
    if num_args > 1:  # We've got arguments
        args = parse()
    else:
        print('Do other stuff')


if __name__ == '__main__':
    main()
