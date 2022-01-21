
"""
This script is for time series prediction based of fft.
Author: Puja Chowdhury
Supervised by: Dr. Austin Downey
Last Updated:Mar 15 17:05:01 2021
updated details: deleting computation time from testdata

"""

#%% Load Libraries
import time as tm
from timeit import default_timer
import IPython as IP
IP.get_ipython().magic('reset -sf')
import numpy as np
import pickle
import math
import numpy as np
from numpy import fft, math
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step

plt.close('all')


#%% Load data
data_pickle = pickle.load(open('nonstationarity_data.pkl', 'rb'))
acceleration=data_pickle['acceleration']
time=data_pickle['time']
#%% Configuration
Ts=(time[-1]-time[0])/time.shape[0]
Fs =math.floor(1/Ts) 
input_time=0.1#training time
time_to_predict=1 # time of prediction
series_length=16.5
sliding_size=1
computation_time=.5
x_test_data = acceleration[int((input_time+computation_time)* Fs):] 
# pickel file nameing
saveNameIdx='1001000500' # input time * 1000,time_to_predict * 1000, computation_time *1000
#%% plot the orginal data and the FFT
plt.figure()
plt.plot(time,acceleration)
plt.xlabel('time')
plt.ylabel('acceleration')
plt.title('original data (time series)')
plt.grid()
#plt.savefig('plots/original_data_time_series')
#%% FFT over original data
N = acceleration.size
T = np.arange(0, N)
P = np.polyfit(T, acceleration, 1)
X_notrend = acceleration - P[0] * T  # detrended x
X_freqdom = fft.fft(X_notrend)  # detrended x in frequency domain
F = fft.fftfreq(N, d=Ts)  # frequencies
plt.figure()
plt.plot(np.absolute(F),np.absolute(X_freqdom), '--',  label='frequency domain')
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.title('original data (frequency domain)')
plt.grid()

#%% data dictionary
td_section = {}  
td_section['time_org']=[]
td_section['x_train_data']=[]
td_section['x_test_data']=[]
td_section['signal_pred_time']=[]
td_section['signal_pred_data']=[]
td_section['mae'] = []
td_section['mse'] = []
td_section['rmse'] = []
td_section['trac'] = []
    
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
    

#%% Process the data

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
    td_section['trac'].append(np.dot(td_section['x_test_data'][select_idx].conj().T, signal_pred)** 2/((np.dot(td_section['x_test_data'][select_idx].conj().T, td_section['x_test_data'][select_idx])) *(np.dot(signal_pred.conj().T, signal_pred))))

    # FFT over extrapolation
    n1 = extrapolation.size
    t2 = np.arange(0, n1)
    p1 = np.polyfit(t2, extrapolation, 1)
    x_notrend1 = extrapolation - p1[0] * t2  # detrended x
    x_freqdom1 = fft.fft(x_notrend1)  # detrended x in frequency domain
    f1 = fft.fftfreq(n1, d=Ts)  # frequencies
      
  # 
    # FFT over signal prediction
    n2 = signal_pred.size
    t3 = np.arange(0, n2)
    p2 = np.polyfit(t3, signal_pred, 1)
    x_notrend2 = signal_pred - p2[0] * t3  # detrended x
    x_freqdom2 = fft.fft(x_notrend2)  # detrended x in frequency domain
    f2 = fft.fftfreq(n2, d=Ts)  # frequencies
      
    
    td_section['signal_pred_data'].append(signal_pred)


print('saving pickle')


# pickle.dump(td_section['time_org'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/time_org'+saveNameIdx+'.pk', 'wb',))
# pickle.dump(td_section['x_train_data'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_train_data'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['x_test_data'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['signal_pred_time'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['signal_pred_data'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['mae'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['mse'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['rmse'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx+'.pk', 'wb'))
# pickle.dump(td_section['trac'], open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx+'.pk', 'wb'))
