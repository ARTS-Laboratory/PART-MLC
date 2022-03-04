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
Y Series Forecasting for Structures Subjected to Nonstationary Inputs
Please cite the following work if you use this code and data:
Chowdhury, P., Conrad, P., Bakos, J. D., & Downey, A. (2021, September). Y Series Forecasting for Structures Subjected to Nonstationary Inputs. In Smart Materials, Adaptive Structures and Intelligent Systems (Vol. 85499, p. V001T03A008). American Society of Mechanical Engineers.

"""

#%% Load Libraries
import numpy as np
from numpy import fft, math

#%% FFT function
def fft_prediction(X,dt,forcast_horizon_steps):
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
    Returns
    -------
    td_section : TYPE
        DESCRIPTION.

    '''
    Y = np.array([*range_with_floats(0,len(X)*dt, dt)])[0:len(X)]
    Ts=(Y[-1]-Y[0])/Y.shape[0] 
    Fs =math.floor(1/Ts) # Sample Rate
    forcast_horizon = math.ceil(forcast_horizon_steps*Ts)
    x_test_data = X[int((forcast_horizon)* Fs):]
    td_section = {"Y_org" : [], "x_train_data" : [], "x_test_data": [], "signal_pred_Y": [], 
                  "signal_pred_data": []}
    # split the data
    count=0
    for i in range_with_floats(0,len(X), forcast_horizon):
        starting_point=int(round(i*Fs))    
        ending_point_train =int(round(starting_point + (forcast_horizon * Fs)))
        ending_point_test = int(round(starting_point + (forcast_horizon * Fs)))
        x_train_data_org = X[starting_point:ending_point_train]
        x_test_data_split = x_test_data[starting_point:ending_point_test]
        if len(x_train_data_org)==0:
            break
        Y_org=Y[starting_point:ending_point_train]
        td_section['Y_org'].append(Y_org)
        td_section['x_train_data'].append(x_train_data_org)
        td_section['x_test_data'].append(x_test_data_split)
        td_section['signal_pred_Y'].append(Y[ending_point_train:ending_point_train+int(round(forcast_horizon*Fs))])
        count+=1

    # Process the data
    for select_idx in range(count):

        x_train_data = td_section['x_train_data'][select_idx]
        # Processing frequency:
        n = x_train_data.size
        t1 = np.arange(0, n)
        p = np.polyfit(t1, x_train_data, 1) # find the trend 
        x_notrend = x_train_data - p[0] * t1  # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n, d=Ts)  # frequencies

        ## sorted  frequencies with more impact on the original FFT

        freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,
                    -20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240]
        fq_idx=[]
        for fq in freq_list:
            if len(np.where(np.isclose(f, fq))[0])==0: # if not able to find any frequencies from freq_list don't append
                pass
            else:
                fq_idx.append(int(np.where(np.isclose(f, fq))[0]))


        n_predict = math.floor(Fs * (forcast_horizon))
        t1 = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t1.size)
        signal_pred_list=[]

        for j in fq_idx:

            ampli = np.absolute(x_freqdom[j]) /n  # find amplitude
            phase = np.angle(x_freqdom[j])  # find phase  
            restored_sig += ampli * np.cos(2 * np.pi * (f[j] / (n / forcast_horizon)) * t1 + phase) # restored signal with respect to phase,amplitude and frequencies

        trend=p[0]
        extrapolation = restored_sig +trend * t1
        signal_pred = extrapolation[int(round((forcast_horizon)* Fs)):int(math.ceil((forcast_horizon + forcast_horizon) * Fs))]    
        signal_pred_list.append(signal_pred)
        td_section['signal_pred_data'].append(signal_pred)
    return td_section

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
        