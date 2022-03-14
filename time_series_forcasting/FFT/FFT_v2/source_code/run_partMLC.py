
"""
Created on Thu Feb 17 01:10:12 2022

@author: Puja Chowdhury

This script is for user. Data input and tuning the configuration 
Published Paper:
Time Series Forecasting for Structures Subjected to Nonstationary Inputs
Please cite the following work if you use this code and data:
Chowdhury, P., Conrad, P., Bakos, J. D., & Downey, A. (2021, September). Time Series Forecasting for Structures Subjected to Nonstationary Inputs. In Smart Materials, Adaptive Structures and Intelligent Systems (Vol. 85499, p. V001T03A008). American Society of Mechanical Engineers.

"""
#%% Load Libraries
import IPython as IP
IP.get_ipython().magic('reset -sf')

import pickle
import partMLC as partMLC
import matplotlib.pyplot as plt

plt.close('all')


# Main Funtion

#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))

X=data_pickle['acceleration']
dt=1.9531228885135136e-05

# User defined parameters
forcast_horizon_steps= 500 # prediction length # here 1s=51200 samples/sec


xx = X[0:5000]

y = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps)





