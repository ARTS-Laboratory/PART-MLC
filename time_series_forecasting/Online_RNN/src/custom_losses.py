import torch

import torch.nn.functional as F


def torch_trac(signal_1, signal_2):
    """ Return TRAC on shifted signals.

        :param torch.Tensor signal_1: Signal to be compared. All values must be positive, nonzero allowed
        :param torch.Tensor signal_2: Signal to be compared. All values must be positive, nonzero allowed

        Assumes signals are centered around 0 with same amplitude
    """
    if signal_1.ndim == 1 and signal_2.ndim == 1:  # If both one dimensional
        # Get absolute minimum of both signals to shift signals to positive
        y_axis_offset = min(torch.min(signal_1), torch.min(signal_2))
        return torch_trac_1d_calc(signal_1 + y_axis_offset, signal_2 + y_axis_offset)
    raise NotImplementedError


def torch_trac_1d_calc(ref_signal: torch.Tensor, predicted_signal: torch.Tensor) -> float:
    """ Implementation of TRAC function.

    :param torch.Tensor ref_signal: Signal to be compared to.
    :param torch.Tensor predicted_signal: Signal to compare.
    :return: TRAC score for given pair of signals.
    :rtype: float
    """
    numerator = torch.dot(ref_signal, predicted_signal)**2
    denominator = torch.dot(ref_signal, ref_signal)*torch.dot(predicted_signal, predicted_signal)
    return numerator/denominator


class MSETRACLoss(torch.nn.Module):
    def __init__(self):
        super(MSETRACLoss, self).__init__()

    def forward(self, inputs, targets):
        mse = F.mse_loss(inputs.view(-1), targets.view(-1))
        trac_score = torch.as_tensor(torch_trac(inputs.view(-1), targets.view(-1)))
        result = mse + (1 - trac_score)
        return result
