import numpy as np


def drop_every_n_point(signal, point):
    """ Drop every nth element from one-dimensional array.

        :param signal: One-dimension array of signal data.
        :type signal: np.ndarray or List
        :param int point: Nth point to remove. (i.e. every 3rd point)
    """
    return np.delete(signal, slice(None, None, point))

