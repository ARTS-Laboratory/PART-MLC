import pandas as pd
import torch

from model_selection.data_prep import get_training_data, get_training_data_sliding_window


def get_toy_data():
    data = pd.DataFrame(
        {'numbers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    return data


def test_get_train_data():
    data = get_toy_data()
    train_d, train_actual = get_training_data(
        data, 0, 5, 4, data_idx='numbers')
    assert torch.equal(
        train_d,
        torch.tensor([[[1]], [[2]], [[3]], [[4]]], dtype=torch.float32))
    assert torch.equal(
        train_actual,
        torch.tensor([[[6]], [[7]], [[8]], [[9]]], dtype=torch.float32))


def test_get_training_data_sliding_window():
    data = get_toy_data()
    train_d, train_actual = get_training_data_sliding_window(
        data, 0, 5, 3, 4, data_idx='numbers')
    assert torch.equal(
        train_d,
        torch.tensor([[[1, 2, 3]], [[2, 3, 4]], [[3, 4, 5]], [[4, 5, 6]]], dtype=torch.float32))
    assert torch.equal(
        train_actual,
        torch.tensor([[[8]], [[9]], [[10]], [[11]]], dtype=torch.float32))
