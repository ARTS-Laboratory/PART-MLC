# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 01:11:27 2022

@author: Puja Chowdhury

This is the whole FFT based prediction function.
This script has three functions:
    1. fft_prediction: FFT based prediction model works
    2.range_with_floats:This function increments the start number by step until it reaches stop.
    link: https://www.dataquest.io/blog/python-range-tutorial/

Published Paper:
Time Series Forecasting for Structures Subjected to Nonstationary Inputs
Please cite the following work if you use this code and data:
Chowdhury, P., Conrad, P., Bakos, J. D., & Downey, A. (2021, September). Time Series Forecasting for Structures Subjected to Nonstationary Inputs. In Smart Materials, Adaptive Structures and Intelligent Systems (Vol. 85499, p. V001T03A008). American Society of Mechanical Engineers.

"""

#%% Load Libraries
import numpy as np
<<<<<<< HEAD:time_series_forcasting/FFT/FFT_Function_PC_version/source_code/partMLC.py
from numpy import fft, math
=======
>>>>>>> 3d7693b12d2430610d82eedd1e4a1238bb44555f:time_series_forcasting/FFT/FFT_Function_PC_V4/source_code/partMLC.py
import warnings

#%% FFT function


def fft_prediction(X,dt,forcast_horizon_steps,freq_list=[],returnVector=True):
    '''
    This function used FFT based model to predict the series data.

    Parameters
    ----------
    X : numpy.ndarray
        This model is taking X data. But any series data can be used instead of X.
    dt : float
        time difference between two sample points.
    forcast_horizon : integer
        length of prediction.
    returnVector: boolean
         Returns the vector of data up to forcast_horizon_steps if returnVector=True
         Just returns 1 point forcast_horizon_steps into the future if returnVector=False
    freq_list: list
        sorted  frequencies with more impact on the original FFT
    Returns
    -------
    Y : List if returnVector==True, else Float
        If returnVector==True, then it will return the predicted vector,
        otherwise it will return the last sample point of the predicted vector.

    '''
     # Set up new time related items. 
    tt = np.array([*range_with_floats(0,len(X)*dt, dt)])[0:len(X)]
    Ts=(tt[-1]-tt[0])/tt.shape[0]
    
    # Detrending and taking FFT
    n = X.size
    n = X.size
    t1 = np.arange(0, n)
    p = np.polyfit(t1, X, 1) # find the trend
    x_notrend = X- p[0] * t1  # detrended x in frequency domain
    
    #%% building a new input length which will be increase or decrease by power of two
    q = 0 # input length  controlling parameter
    if n< int(1/Ts):
        while (n - int((2 ** q) * (1 / Ts))) < 0:
            q-=1
            if q<-1:
                warnings.warn("The input length must be greater than the half of the sample rate for better prediction")
    elif n> int(1/Ts):
        while (n - int((2 ** q) * (1 / Ts))) > int((2 ** q) * (1 / Ts)):
            q+=1
    else:
        pass
    n_prime = int((2 ** q) * (1 / Ts)) # new input length
    x_freqdom = fft.fft(x_notrend[0:n_prime],n=n_prime) # detrended x in frequency domain
    f = fft.fftfreq(len(x_freqdom), d=Ts)  # frequencies
    
   # build the index of freqency values
    freq_idx=[]
    if freq_list==[]:
        warnings.warn("No frequency list has been provided. Running for all the frequencies.")
        freq_list = f #If no frequencies provided, it will return a warning and use all frequences.
        freq_idx = list(range(len(freq_list)))
    else:
        for fq in freq_list:
            a = np.isclose(f, fq,atol=0.1) 
            b = np.where(a)[0] # find true indcies 
            freq_idx.append(int(b))

    # creates vecotrs for rebuilding the signal
    t2 = np.arange(0, n + forcast_horizon_steps)
    restored_sig = np.zeros(t2.size)
    # rebuild the signal
    for j in freq_idx:
        ampli = np.absolute(x_freqdom[j]) /n_prime  # find amplitude
        phase = np.angle(x_freqdom[j])  # find phase
        freq_rad_per_sec = 2 * np.pi * (f[j]) # report the frequency in rad for the cos function
        freq_rad_per_sample = freq_rad_per_sec*Ts # returns frequency in rad_per_sample.Reason: f[j]*Ts: to make sampling frequency cause only f[j] is just frequency
        restored_sig += ampli * np.cos(freq_rad_per_sample * t2 + phase) # restored signal with respect to phase,amplitude and frequencies
   # add trend
    trend=p[0]
    Y = restored_sig +trend * t2
    
    if returnVector:
        return Y[len(X):]
    else:
        return Y[-1]

    
#%% range function with floats
def range_with_floats(start, stop, step):
    '''
    This function increments the start number by step until it reaches stop.
    link: https://www.dataquest.io/blog/python-range-tutorial/
    
    Parameters
    ----------
    start : float
        Starting point.
    stop : float
        Stopping point.
    step : float
        Step size.

    Yields
    ------
    start : float
        Generate range-like returns with floats.

    '''
    while stop > start:
        yield start
        start += step
<<<<<<< HEAD:time_series_forcasting/FFT/FFT_Function_PC_version/source_code/partMLC.py
=======
        
>>>>>>> 3d7693b12d2430610d82eedd1e4a1238bb44555f:time_series_forcasting/FFT/FFT_Function_PC_V4/source_code/partMLC.py
