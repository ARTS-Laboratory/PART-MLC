
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

# Main Funtion

#%% Load data
data_pickle = pickle.load(open('../data/nonstationarity_data.pkl', 'rb'))
X=data_pickle['acceleration']
dt=1.9531228885135136e-05

# User defined parameters
#freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,-20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240]# sorted  frequencies with more impact on the original FFT

forcast_horizon_steps = 5000 # prediction length # here 1s=51200 samples/sec

# Input length should capture the minimum frequency. 
xx_length = 30000
xx = X[0:xx_length]

# window = np.hanning(xx.shape[0])
# xx = xx *window

#%% plot the results using all freqs
y1_withoutFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,returnVector=True)# Returns the vector of data up to forcast_horizon_steps if returnVector=True

# plot the code
forcast = np.hstack((np.full([xx_length], np.nan),y1_withoutFreq))


truth = X[xx.shape[0]:forcast.shape[0]]
error = y1_withoutFreq - truth


SNR_db = 10*np.log10(np.sqrt(np.mean(truth**2))/np.sqrt(np.mean(error**2)))**2



plt.figure()
plt.title('Without Freq with SNR$_{db}$ of ' + str(np.round(SNR_db,2)))
plt.plot(X[0:forcast.shape[0]],color='gray',label='truth')
plt.plot(forcast,':',label='forcast')
plt.plot(xx,'--',label='training data')
plt.xlabel('time (data points)')
plt.ylabel('acceleration (g)')
#plt.xlim([45000,58000])
plt.legend()
plt.tight_layout()






#%% plot the results using all freqs

snr_list = []
xx_length_list = []


forcast_horizon_steps = 5000 # prediction length # here 1s=51200 samples/sec

# Input length should capture the minimum frequency. 
for xx_length in range (5000,70000,2500):
    xx = X[0:xx_length]
    
    # window = np.hanning(xx.shape[0])
    # xx = xx *window
    
    #%% plot the results using all freqs
    y1_withoutFreq = prediction_signal=partMLC.fft_prediction(xx,dt,forcast_horizon_steps,returnVector=True)# Returns the vector of data up to forcast_horizon_steps if returnVector=True
    
    # plot the code
    forcast = np.hstack((np.full([xx_length], np.nan),y1_withoutFreq))
    
    
    truth = X[xx.shape[0]:forcast.shape[0]]
    error = y1_withoutFreq - truth
    
    
    SNR_db = 10*np.log10(np.sqrt(np.mean(truth**2))/np.sqrt(np.mean(error**2)))**2
    snr_list.append(SNR_db)
    xx_length_list.append(xx.shape[0])



plt.figure()
plt.plot(xx_length_list,snr_list,'-o')
plt.title('SNR$_{\mathrm{db}}$ vs size of input data')
plt.xlabel('data length (data points)')
plt.ylabel(r'SNR$_{\mathrm{db}}$')
plt.tight_layout()
































