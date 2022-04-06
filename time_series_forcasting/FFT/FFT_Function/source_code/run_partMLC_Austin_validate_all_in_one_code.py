
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
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


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
        
        
# Main Funtion

#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))
XX=data_pickle['acceleration']
dt=1.9531228885135136e-05

# User defined parameters
#freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,-20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240]# sorted  frequencies with more impact on the original FFT
freq_list=[]

forcast_horizon_steps = 5000 # prediction length # here 1s=51200 samples/sec

# Input length should capture the minimum frequency. 
xx_length = 50000 #51200
xx = XX[0:xx_length]


# window = np.hanning(xx.shape[0])
# xx = xx *window


X = xx

import warnings as warnings

#%% Copy over the function

# Set up new time related items. 
tt = np.array([*range_with_floats(0,len(X)*dt, dt)])[0:len(X)]
Ts=(tt[-1]-tt[0])/tt.shape[0]
forcast_horizon = np.math.ceil(forcast_horizon_steps*Ts) # Return the ceiling of the input, element-wise.
#forcast_horizon = forcast_horizon_steps*Ts # Return the ceiling of the input, element-wise.


# Detrending and taking FFT
n = X.size
t1 = np.arange(0, n)
p = np.polyfit(t1, X, 1) # find the trend
x_notrend = X - p[0] * t1  # detrended x
x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
f = np.fft.fftfreq(n, d=Ts)  # frequencies

# # If no frequencies provided, return a warning and use all frequences.
# freq_idx=[]
# if freq_list==[]:
#     warnings.warn("No frequency list has been provided. Running for all the frequencies.")
#     freq_list = list(f) 

# else:
#     for i in freq_list:
#         a = np.where(np.isclose(f, i,atol=0.1))[0]
#         freq_idx.append(int(a)) # I don't like this code.       

    

freq_idx=[]
if freq_list==[]:
    warnings.warn("No frequency list has been provided. Running for all the frequencies.")
    freq_list = f #If no frequencies provided, it should return an error. Or, use all frequences and return a warning.
    freq_idx = list(range(len(freq_list)))

else:
    for fq in freq_list:
        a = np.isclose(f, fq,atol=0.1)
        b = np.where(a)[0] # find true indcies 
        freq_idx.append(int(b))
            
    
    
    
    
# for fq in freq_list:
#     if len(np.where(np.isclose(f, fq,atol=0.1))[0])==0: # if not able to find any frequencies from freq_list don't append
#         pass
#     else:
#         freq_idx.append(int(np.where(np.isclose(f, fq,atol=0.1))[0]))
        
        
# # build the index of frequencies 
# freq_idx=[]    
# for i in freq_list:
#     if len(np.where(np.isclose(f, i,atol=0.1))[0])==0: # if not able to find any frequencies from freq_list don't append
#         pass
  
#     else:
#         a = np.isclose(f, i,atol=0.1))[0]
#         freq_idx.append(int(np.where(a)) # I don't like this code. 


# creates vecotrs for rebuilding the signal
t2 = np.arange(0, n + forcast_horizon_steps)
restored_sig = np.zeros(t2.size)

# 
for j in freq_idx:
    ampli = np.absolute(x_freqdom[j]) /n  # find amplitude
    phase = np.angle(x_freqdom[j])  # find phase
    freq_rad_per_sec = 2 * np.pi * (f[j]) # report the frequency in rad for the cos function
    freq_rad_per_sample = freq_rad_per_sec / (n / forcast_horizon) # returns frequency in rad_per_sample. I think the forcast horizion should not be there.
    restored_sig += ampli * np.cos(freq_rad_per_sample * t2 + phase) # restored signal with respect to phase,amplitude and frequencies

trend=p[0]
Y = restored_sig + trend * t2

# if returnVector:
#     return Y[len(X):]
# else:
#     return Y[-1]

#%% plot the results using all freqs
#y1_withoutFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,returnVector=True)# Returns the vector of data up to forcast_horizon_steps if returnVector=True

y1_withoutFreq = Y[len(X):]


# plot the code
forcast = np.hstack((np.full([xx_length], np.nan),y1_withoutFreq))


plt.figure()
plt.title('Without Freq')
plt.plot(XX[0:forcast.shape[0]],color='gray',label='truth')
plt.plot(forcast,':',label='forcast')
plt.plot(xx,'--',label='training data')
plt.xlabel('time (data points)')
plt.ylabel('acceleration (g)')
plt.xlim([45000,58000])
plt.legend()
plt.tight_layout()










# #%% plot the results using all freqs
# y1_withFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,freq_list,returnVector=True) # Returns the vector of data up to forcast_horizon_steps if returnVector=True

# # plot the code
# forcast = np.hstack((np.full([xx_length], np.nan),y1_withFreq))

# plt.figure()
# plt.title('With Freq')
# plt.plot(X[0:forcast.shape[0]],color='gray',label='truth')
# plt.plot(forcast,':',label='forcast')
# plt.plot(xx,'--',label='training data')
# plt.xlabel('time (data points)')
# plt.ylabel('acceleration (g)')
# plt.legend()
# plt.tight_layout()









































