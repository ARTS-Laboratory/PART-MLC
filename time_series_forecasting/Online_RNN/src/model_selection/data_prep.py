import torch


def get_training_data(data, start_idx, prediction_gap, length, data_idx, dtype=torch.float32):
    """ Return tuple of training data.

        :param data: Pandas data array to pull data from.
        :type data: pd.DataFrame
        :param int start_idx: Index to start collecting data from.
        :param int prediction_gap: How many points into the future
         model should predict.
        :param int length: Total number of points to put in training arrays.
        :param data_idx: Column index of data to pull values.
        :returns: Tuple of training data and expected values.
        :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    train_data = torch.tensor(
        data.loc[start_idx:start_idx + length - 1, data_idx].array,
        dtype=dtype).view(-1, 1, 1)
    train_actual = torch.tensor(
        data.loc[
            start_idx + prediction_gap:
            start_idx + prediction_gap + length - 1,
            data_idx].array, dtype=dtype).view(-1, 1, 1)
    return train_data, train_actual
