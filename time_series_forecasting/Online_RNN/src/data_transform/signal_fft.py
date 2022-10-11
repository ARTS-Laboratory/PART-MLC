import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq


def sampling_frequency(time):
    """ """
    return len(time)/(time[-1] - time[0])


def plot_fft(signal, samp_space):
    fig = plt.figure()
    length = len(signal)
    yf = fft(signal)
    xf = fftfreq(length, samp_space)[:length//2]
    plt.semilogy(xf, 2.0/length * np.abs(yf[0:length//2]))
    plt.grid()
    return fig


def main():
    pass

if __name__ == '__main__':
    main()