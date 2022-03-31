
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

# clear out the console and remove all variables present
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import pickle
import partMLC as partMLC
import matplotlib.pyplot as plt
plt.close('all')


# Main Funtion

#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))

X=data_pickle['acceleration'][900000::]
dt=1.9531228885135136e-05

# User defined parameters
freq_list = [20,60,70,80,100,120,140,150,160,170,180,-20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180]# sorted  frequencies with more impact on the original FFT
forcast_horizon_steps= 5120 # prediction length # here 1s=51200 samples/sec

# Input length should capture the minimum frequency. 

xx = X[0:5120]

y1_withoutFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,True)# Returns the vector of data up to forcast_horizon_steps if returnVector=True
y1_withFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,True, freq_list) # Returns the vector of data up to forcast_horizon_steps if returnVector=True
y2_withoutFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,False)# Just returns 1 point forcast_horizon_steps into the future if returnVector=False
y2_withFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,False, freq_list)# Just returns 1 point forcast_horizon_steps into the future if returnVector=False







