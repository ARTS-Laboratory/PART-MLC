
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
import partMLC as partMLC

# Main Funtion
# def main():
#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))
X=data_pickle['acceleration']
decision_for_dt = input("Here are the options \n 1. dt from X \n 2. Y from data \n Choose your Option: ")

if int(decision_for_dt) == 1:
    dt=1.9531228885135136e-05
    Y = np.array([*range_with_floats(0,len(X)*dt, dt)])[0:len(X)]
else:
    Y=data_pickle['time']


# User defined parameters
window_steps= 51200# Learning Window Length in seconds
forcast_horizon_steps= 51200 # time of prediction
sliding_steps= 51200 # rolling window size
result_dict = partMLC.fft(X, Y, window_steps, forcast_horizon_steps, sliding_steps)

#%% Merging data
Sig_pred_Y_all = np.concatenate(result_dict["signal_pred_Y"], axis=0, out=None)
signal_pred_data_all = np.concatenate(result_dict["signal_pred_data"], axis=0, out=None)
x_test_data_all = np.concatenate(result_dict["x_test_data"], axis=0, out=None)

# Plotting
dt=(Y[-1]-Y[0])/Y.shape[0] 
Fs =math.floor(1/dt) # Sample Rate
plotting_all(Y, X, dt, Fs, Sig_pred_Y_all, x_test_data_all,signal_pred_data_all)

# if __name__ == '__main__':
#     main()





































