# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 01:11:27 2022

@author: chypu
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

#%% FFT function
def fft_prediction(acceleration, time, Ts, Fs, input_time, time_to_predict, series_length, sliding_size, computation_time):
    '''
    This function used FFT based model to predict the series data.

    Parameters
    ----------
    acceleration : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.
    Ts : TYPE
        DESCRIPTION.
    Fs : TYPE
        DESCRIPTION.
    input_time : TYPE
        DESCRIPTION.
    time_to_predict : TYPE
        DESCRIPTION.
    series_length : TYPE
        DESCRIPTION.
    sliding_size : TYPE
        DESCRIPTION.
    computation_time : TYPE
        DESCRIPTION.

    Returns
    -------
    td_section : TYPE
        DESCRIPTION.

    '''

    
    x_test_data = acceleration[int((input_time+computation_time)* Fs):]
    td_section = {"time_org" : [], "x_train_data" : [], "x_test_data": [], "signal_pred_time": [], 
                  "signal_pred_data": [], "mae": [], "mse": [], "rmse": []}
    # split the data
    count=0
    for i in range_with_floats(0,series_length, sliding_size):
        # starting_point=round(i*Fs)
        starting_point=int(round(i*Fs))    
        ending_point_train =int(round(starting_point + (input_time * Fs)))
        ending_point_test = int(round(starting_point + (time_to_predict * Fs)))
        x_train_data_org = acceleration[starting_point:ending_point_train]
        x_test_data_split = x_test_data[starting_point:ending_point_test]

        time_org=time[starting_point:ending_point_train]
        td_section['time_org'].append(time_org)
        td_section['x_train_data'].append(x_train_data_org)
        td_section['x_test_data'].append(x_test_data_split)
    #   td_section['train+pred_time'].append(time[starting_point:ending_point_train+round(time_to_predict*Fs)])
        td_section['signal_pred_time'].append(time[ending_point_train+int(round(computation_time*Fs)):ending_point_train+int(round(computation_time*Fs))+int(round(time_to_predict*Fs))])
        count+=1

    # Process the data

    for select_idx in range(count):


        x_train_data = td_section['x_train_data'][select_idx]
        # Processing frequency:
        n = x_train_data.size
        t1 = np.arange(0, n)
        p = np.polyfit(t1, x_train_data, 1)
        x_notrend = x_train_data - p[0] * t1  # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n, d=Ts)  # frequencies

        ## sorted  frequencies with more impect on the original FFT

        freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,
                    -20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240]
        fq_idx=[]
        for fq in freq_list:
            # fq_idx.append(np.where(np.isclose(f, fq)))
            if len(np.where(np.isclose(f, fq))[0])==0: # if not able to find any frequencies from freq_list don't append
                pass
                # print(fq)
            else:
                fq_idx.append(int(np.where(np.isclose(f, fq))[0]))


        n_predict = math.floor(Fs * (time_to_predict+computation_time))
        t1 = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t1.size)
        signal_pred_list=[]

        for j in fq_idx:

            ampli = np.absolute(x_freqdom[j]) /n  # amplitude
            phase = np.angle(x_freqdom[j])  # phase  
            restored_sig += ampli * np.cos(2 * np.pi * (f[j] / (n / input_time)) * t1 + phase)

        trend=p[0]
        extrapolation = restored_sig +trend * t1
        # signal_pred = extrapolation[int(round(input_time * Fs)):int(math.ceil((input_time + time_to_predict) * Fs))] #  second prediction # round make decimal points a round number, math.ceil get the higher round number
        signal_pred = extrapolation[int(round((input_time+computation_time)* Fs)):int(math.ceil((input_time +computation_time+ time_to_predict) * Fs))]    
        signal_pred_list.append(signal_pred)
        ## error calculation
        td_section['mae'].append(mean_absolute_error(td_section['x_test_data'][select_idx], signal_pred))
        td_section['mse'].append(sklearn.metrics.mean_squared_error(td_section['x_test_data'][select_idx], signal_pred))
        td_section['rmse'].append(math.sqrt(sklearn.metrics.mean_squared_error(td_section['x_test_data'][select_idx], signal_pred)))   
        td_section['signal_pred_data'].append(signal_pred)
    return td_section

#%% Plotting function
def plotting_all(time, acceleration, Ts, Fs, Sig_pred_time_all, x_test_data_all,signal_pred_data_all):
    '''
    This function will plots the original data and the output of the models with error analysis.

    Parameters
    ----------
    time : numpy.ndarray
        time data.
    acceleration : numpy.ndarray
        This model is taking acceleration data. But any series data can be used instead of acceleration.
    Ts : float
        Sampling time.
    Fs : float
        Sampling rate in Hz.
    Sig_pred_time_all : numpy.ndarray
        DESCRIPTION.
    x_test_data_all : numpy.ndarray
        DESCRIPTION.
    signal_pred_data_all : numpy.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # plotting with specific parameters
    # Reseting plot parameter to default
    plt.rcdefaults()
    # Updating Parameters for Paper
    params = {
        'lines.linewidth' :1,
        'lines.markersize' :2,
       'axes.labelsize': 8,
        'axes.titlesize':8,
        'axes.titleweight':'normal',
        'font.size': 8,
        'font.family': 'Times New Roman', # 'Times New RomanArial'
        'font.weight': 'normal',
        'mathtext.fontset': 'stix',
        'legend.shadow':'False',
       'legend.fontsize': 6,
       'xtick.labelsize':8,
       'ytick.labelsize': 8,
       'text.usetex': False,
        'figure.autolayout': True,
       'figure.figsize': [10,10] # width and height in inch (max width = 7.5", max length = 10")
       }
    plt.rcParams.update(params)

    fig1 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)

    #plot the orginal data and the FFT
    ax1 = fig1.add_subplot(spec2[0, 0])
    ax1.plot(time,acceleration,'-',linewidth=.5, label='original data')
    ax1.set_title('Original Data', y=-.4, ha='center')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s$^2$)')
    # plt.xlim(0,1.5)
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

    # FFT over original data
    N = acceleration.size
    T = np.arange(0, N)
    P = np.polyfit(T, acceleration, 1)
    X_notrend = acceleration - P[0] * T  # detrended x
    X_freqdom = fft.fft(X_notrend)  # detrended x in frequency domain
    F = fft.fftfreq(N, d=Ts)  # frequencies

    ax2 = fig1.add_subplot(spec2[0, 1])
    ax2.plot(np.absolute(F),np.absolute(X_freqdom), '--',  label='frequency domain')
    ax2.set_title('FFT Plot', y=-.4, ha='center')
    plt.xlabel('frequency (s)')
    plt.ylabel('amplitude')
    # plt.xlim(0,1.5)
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)


    # prediction and truth plot
    ax3 = fig1.add_subplot(spec2[1, 0])
    ax3.plot(Sig_pred_time_all,x_test_data_all,'-',linewidth=.5, label='data')
    ax3.plot(Sig_pred_time_all,signal_pred_data_all,'--',linewidth=.5, label='predicted')
    # plt.xlabel('time (s)\n (a)')
    # ax.set_title("Hello Title", y=0, pad=-35, verticalalignment="bottom")
    ax3.set_title('Prediction and Truth', y=-.5, ha='center')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s$^2$)')
    # plt.xlim(0,1.5)
    plt.grid()
    plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

    # Error plot

    ax4 = fig1.add_subplot(spec2[1, 1])
    ax4.plot(Sig_pred_time_all,abs(x_test_data_all-signal_pred_data_all),'-',linewidth=.5, label='error')#label= 'error for 0.01 sec learning window'
    ax4.set_title('Error', y=-.5, ha='center')
    plt.xlabel('time (s)')
    plt.ylabel('error (m/s$^2$)')
    # plt.xlim(0,1.5)
    plt.grid()
    # plt.legend()
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
        