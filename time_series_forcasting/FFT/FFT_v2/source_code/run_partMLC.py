
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
import pickle
import partMLC as partMLC

# Main Funtion
#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))
X=data_pickle['acceleration']
dt=1.9531228885135136e-05
# User defined parameters
forcast_horizon_steps= 51200 # prediction length # here 1s=51200 samples/sec
result_dict = partMLC.fft_prediction(X,dt,forcast_horizon_steps)
