# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:25:20 2021

@author: chypu
"""
#%% Load Libraries
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

#%% plotting with specific parameters
# Reseting plot parameter to default
plt.rcdefaults()
# Updating Parameters for Paper
params = {
    'lines.linewidth' :.2,
    'lines.markersize' :1,
   'axes.labelsize': 8,
    'axes.titlesize':8,
    'axes.titleweight':'normal',
    'font.size': 8,
    'font.family': 'Times New Roman', # 'Times New RomanArial'
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'legend.shadow':'False',
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'text.usetex': False,
    'figure.autolayout': True,
   'figure.figsize': [6.5,2.5] # width and height in inch (max width = 7.5", max length = 10")
   }
plt.rcParams.update(params)
#%% Load data
data_pickle = pickle.load(open('../data/3_data_sets/testing data_cut_data/nonstationarity_data.pkl', 'rb'))
acceleration=data_pickle['acceleration']
time=data_pickle['time']
# %% plot the orginal data and the FFT
plt.figure()
fig1 = plt.figure()

ax1 = fig1.add_subplot(211)
ax1.plot(time,acceleration)
ax1.plot([0,0],[-.15,.15],'--',color='black',linewidth=1)
ax1.plot([.4,.4,],[-.15,.15],'-',color='black',linewidth=1)
ax1.plot([-.4,-.4],[-.15,.15],'-',color='black',linewidth=1)
plt.xlim(-8,8)
plt.ylim(-.15,.15)
plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
# plt.title('original data (time series)')
plt.grid()
ax1 = fig1.add_subplot(212)
ax1.plot(time,acceleration,linewidth=.5)
ax1.plot([0,0],[-.15,.15],'--',color='black',linewidth=1)

plt.xlim(-.5,.5)
plt.ylim(-.15,.15)
plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
# plt.title('original data (time series)')
plt.grid()
fig1.tight_layout()
plt.savefig('plots/original_data_time_series', dpi=400)
plt.savefig('plots/original_data_time_series.pdf',dpi=400)