""" Thompson 2022"""

import numpy as np

from numpy import ndarray
from scipy.signal import periodogram


def root_mean_square_frequency(vector: ndarray, time_step: float) -> float:
    """ Return root mean square frequency."""
    frequencies, power_spectrum = periodogram(vector, 1/time_step, scaling='spectrum')
    return np.sqrt(np.sum(frequencies**2 * power_spectrum, axis=-1)
                   / np.sum(power_spectrum, axis=-1))


def trac(signal_1, signal_2):
    """ Return TRAC on shifted signals.

        :param signal_1: Signal to be compared. All values must be positive, nonzero allowed
        :type signal_1: np.ndarray
        :param signal_2: Signal to be compared. All values must be positive, nonzero allowed
        :type signal_2: np.ndarray

        Assumes signals are centered around 0 with same amplitude
    """
    if signal_1.ndim == 1 and signal_2.ndim == 1:  # If both one dimensional
        # Get absolute minimum of both signals to shift signals to positive
        y_axis_offset = min(np.min(signal_1), np.min(signal_2))
        return trac_1d_calc(signal_1 + y_axis_offset, signal_2 + y_axis_offset)
    raise NotImplementedError


def trac_1d_calc(ref_signal, predicted_signal):
    """ Implementation of TRAC function."""
    numerator = np.dot(ref_signal, predicted_signal)**2
    denominator = np.dot(ref_signal, ref_signal)*np.dot(predicted_signal, predicted_signal)
    return numerator/denominator


def snr(ref_signal, predicted_signal, sampling_frequency):
    """ Implementation of signal-to_noise ratio"""
    ref = root_mean_square_frequency(ref_signal, sampling_frequency)
    pred = root_mean_square_frequency(predicted_signal - ref_signal, sampling_frequency)
    return 20 * np.log10(ref/pred)

