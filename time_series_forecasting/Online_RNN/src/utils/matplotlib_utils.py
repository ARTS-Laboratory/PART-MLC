""" Thompson 2022"""
import os
import matplotlib.pyplot as plt


def save_fig(figure: plt.Figure, filename, folder=None, save_args=None, save_kwargs=None):
    """ Save Matplotlib figure to file path(s).

    :param figure: Matplotlib figure to save
    :type figure: plt.Figure or int
    :param str filename: Path to save location
    :param folder: Folder path to prefix filename
    :type folder: str or None
    :param save_args: arguments to pass to plt.savefig()
    :param save_kwargs: key-word arguments to pass to plt.savefig()


    """
    new_dir = folder if folder is not None else ''
    if isinstance(filename, str):
        args = list() if save_args is None else save_args
        kwargs = dict() if save_kwargs is None else save_kwargs
        figure.savefig(
            os.path.join(new_dir, filename),
            *args, **kwargs)
    elif isinstance(filename, list):
        raise NotImplementedError('Code not complete')
        for name in filename:
            figure.savefig()
