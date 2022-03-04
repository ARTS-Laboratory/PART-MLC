# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 01:11:27 2022

@author: Puja Chowdhury

This is the whole FFT based prediction function.
This script has three functions:
    1. partMLC: FFT based prediction model works
    2. plotting_all: this function is for plotting output of the model
    3.range_with_floats:This function increments the start number by step until it reaches stop.
    link: https://www.dataquest.io/blog/python-range-tutorial/

Published Paper:
Time Series Forecasting for Structures Subjected to Nonstationary Inputs
Please cite the following work if you use this code and data:
Chowdhury, P., Conrad, P., Bakos, J. D., & Downey, A. (2021, September). Y Series Forecasting for Structures Subjected to Nonstationary Inputs. In Smart Materials, Adaptive Structures and Intelligent Systems (Vol. 85499, p. V001T03A008). American Society of Mechanical Engineers.

"""

#%% Load Libraries
import IPython as IP
IP.get_ipython().magic('reset -sf')
import numpy as np
from numpy import fft, math
import sklearn
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

#%% FFT function
def partMLC(X, Y, window_steps, forcast_horizon_steps, sliding_steps):
    '''
    This function used FFT based model to predict the series data.

    Parameters
    ----------
    X : numpy.ndarray
        This model is taking X data. But any series data can be used instead of X.
    Y : numpy.ndarray
        Y data.
    window_steps : int
        learning window length.
    forcast_horizon : integer
        length of prediction.
    sliding_size : integer
       window sliding size which should be same as Y of prediction.
    
    Returns
    -------
    td_section : TYPE
        DESCRIPTION.

    '''

    Ts=(Y[-1]-Y[0])/Y.shape[0] 
    Fs =math.floor(1/Ts) # Sample Rate
    window_length = math.ceil(window_steps*Ts)
    forcast_horizon = math.ceil(forcast_horizon_steps*Ts)
    sliding_size = math.ceil(sliding_steps*Ts)
    x_test_data = X[int((window_length)* Fs):]
    td_section = {"Y_org" : [], "x_train_data" : [], "x_test_data": [], "signal_pred_Y": [], 
                  "signal_pred_data": []}
    # split the data
    count=0
    for i in range_with_floats(0,len(X), sliding_size):
        starting_point=int(round(i*Fs))    
        ending_point_train =int(round(starting_point + (window_length * Fs)))
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
            restored_sig += ampli * np.cos(2 * np.pi * (f[j] / (n / window_length)) * t1 + phase) # restored signal with respect to phase,amplitude and frequencies

        trend=p[0]
        extrapolation = restored_sig +trend * t1
        signal_pred = extrapolation[int(round((window_length)* Fs)):int(math.ceil((window_length + forcast_horizon) * Fs))]    
        signal_pred_list.append(signal_pred)
        td_section['signal_pred_data'].append(signal_pred)
    return td_section

#%% Plotting function
def plotting_all(Y, X, Ts, Fs, Sig_pred_Y_all, x_test_data_all,signal_pred_data_all):
    '''
    This function will plots the original data and the output of the models with error analysis.

    Parameters
    ----------
    Y : numpy.ndarray
        Y data.
    X : numpy.ndarray
        This model is taking X data. But any series data can be used instead of X.
    Ts : float
        Sampling Y.
    Fs : float
        Sampling rate in Hz.
    Sig_pred_Y_all : numpy.ndarray
       Y of  predicted signal.
    x_test_data_all : numpy.ndarray
        Truth value/ original data.
    signal_pred_data_all : numpy.ndarray
       predicted signal.

    Returns
    -------
    None.

    '''
    # plotting with specific parameters
    # Reseting plot parameter to default
    plt.rcdefaults()
    params = {
        'lines.linewidth' :1,
        'lines.markersize' :2,
       'axes.labelsize': 8,
        'axes.titlesize':8,
        'axes.titleweight':'normal',
        'font.size': 8,
        'font.family': 'Ys New Roman', # 'Ys New RomanArial'
        'font.weight': 'normal',
        'mathtext.fontset': 'stix',
        'legend.shadow':'False',
       'legend.fontsize': 6,
       'xtick.labelsize':8,
       'ytick.labelsize': 8,
       'text.usetex': False,
        'figure.autolayout': True,
       'figure.figsize': [6.5,6.5] 
       }
    plt.rcParams.update(params)

    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)

    #plot the orginal data 
    ax1 = fig1.add_subplot(spec2[0, 0])
    ax1.plot(Y,X,'-',linewidth=.5, label='original data')
    ax1.set_title('Original Data', y=1, ha='center')
    plt.xlabel('Y (s)')
    plt.ylabel('X (m/s$^2$)')
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

    # FFT over original data
    N = X.size
    T = np.arange(0, N)
    P = np.polyfit(T, X, 1) # find trend
    X_notrend = X - P[0] * T  # detrended x
    X_freqdom = fft.fft(X_notrend)  # detrended x in frequency domain
    F = fft.fftfreq(N, d=Ts)  # frequencies

    ax2 = fig1.add_subplot(spec2[0, 1])
    ax2.plot(np.absolute(F),np.absolute(X_freqdom), '--',  label='frequency domain')
    ax2.set_title('FFT Plot', y=1, ha='center')
    plt.xlabel('frequency (s)')
    plt.ylabel('amplitude')
    plt.grid()
    plt.legend(loc = "upper right",facecolor='white', edgecolor = 'black', framealpha=1)


    # prediction and truth plot
    ax3 = fig1.add_subplot(spec2[1, 0])
    ax3.plot(Sig_pred_Y_all,x_test_data_all,'-',linewidth=.5, label='truth')
    ax3.plot(Sig_pred_Y_all,signal_pred_data_all[0:len(x_test_data_all)],'--',linewidth=.5, label='predicted')
    # plt.xlabel('Y (s)\n (a)')
    # ax.set_title("Hello Title", y=0, pad=-35, verticalalignment="bottom")
    ax3.set_title('Prediction and Truth', y=1, ha='center')
    plt.xlabel('Y (s)')
    plt.ylabel('X (m/s$^2$)')
    # plt.xlim(-1.2,1.2)
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

    # Error plot
    ax4 = fig1.add_subplot(spec2[1, 1])
    ax4.plot(Sig_pred_Y_all,abs(x_test_data_all-signal_pred_data_all[0:len(x_test_data_all)]),'-',linewidth=.5, label='error')#label= 'error for 0.01 sec learning window'
    ax4.set_title('Error', y=1, ha='center')
    plt.xlabel('Y (s)')
    plt.ylabel('error (m/s$^2$)')
    # plt.xlim(-1.2,1.2)
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

    plt.savefig('../plots/output_plot.png', dpi=400)
    plt.savefig('../plots/output_plot.pdf', dpi=400)

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
        
