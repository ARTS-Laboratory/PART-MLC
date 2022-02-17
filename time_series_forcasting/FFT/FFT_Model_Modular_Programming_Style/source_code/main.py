
"""
Created on Thu Feb 17 01:10:12 2022

@author: Puja Chowdhury

This script is for user. Data input and tuning the configuration and plot the output by this one.

Published Paper:
Time Series Forecasting for Structures Subjected to Nonstationary Inputs
Please cite the following work if you use this code and data:
Chowdhury, P., Conrad, P., Bakos, J. D., & Downey, A. (2021, September). Time Series Forecasting for Structures Subjected to Nonstationary Inputs. In Smart Materials, Adaptive Structures and Intelligent Systems (Vol. 85499, p. V001T03A008). American Society of Mechanical Engineers.


"""
#%% Load Libraries
import IPython as IP
IP.get_ipython().magic('reset -sf')
import numpy as np
import pickle
from numpy import math
from model import *

# Main Funtion
def main():
    #%% Load data
    data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))
    acceleration=data_pickle['acceleration']
    time=data_pickle['time']
    Ts=(time[-1]-time[0])/time.shape[0]
    Fs =math.floor(1/Ts)
    
    # User defined parameters
    input_time=1# Learning Window Length in seconds
    time_to_predict=1 # time of prediction
    series_length=16.5 # Length of the series data
    sliding_size=1 # window sliding size which should be same as time of prediction
    computation_time=.1 # assuming computationa time.
    result_dict = fft_prediction(acceleration, time, Ts, Fs, input_time, time_to_predict, series_length, sliding_size, computation_time)
    
    #%% Merging data
    Sig_pred_time_all = np.concatenate(result_dict["signal_pred_time"], axis=0, out=None)
    signal_pred_data_all = np.concatenate(result_dict["signal_pred_data"], axis=0, out=None)
    x_test_data_all = np.concatenate(result_dict["x_test_data"], axis=0, out=None)
    
    # Plotting
    plotting_all(time, acceleration, Ts, Fs, Sig_pred_time_all, x_test_data_all,signal_pred_data_all)

if __name__ == '__main__':
    main()