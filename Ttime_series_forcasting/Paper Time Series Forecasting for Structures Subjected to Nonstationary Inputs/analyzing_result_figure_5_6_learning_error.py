# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:38:40 2021

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
import matplotlib.gridspec as gridspec
import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
#%% learn .01 s, pred 1s
saveNameIdx1='101000100' 
x_test_data_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx1+'.pk', 'rb'))
signal_pred_time_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx1+'.pk', 'rb'))
signal_pred_data_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx1+'.pk', 'rb'))
mae_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx1+'.pk', 'rb'))
mse_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx1+'.pk', 'rb'))
rmse_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx1+'.pk', 'rb'))
trac_01=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx1+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_01 = np.concatenate(signal_pred_time_01, axis=0, out=None)
signal_pred_data_all_01 = np.concatenate(signal_pred_data_01, axis=0, out=None) 
# time_org_all_1 = np.concatenate(time_org_1, axis=0, out=None)
# x_train_data_all_1 = np.concatenate(x_train_data_1, axis=0, out=None) 
x_test_data_all_01 = np.concatenate(x_test_data_01, axis=0, out=None)
mae_all_01 = (mae_01)
mse_all_01=(mse_01)
rmse_all_01 = (rmse_01)
trac_all_01=(trac_01)

#%% learn .02 s, pred 1s
saveNameIdx2='201000100' 
# time_org_01=pickle.load(open('time_org'+saveNameIdx2+'.pk','rb'))
# x_train_data_01=pickle.load (open('x_train_data'+saveNameIdx2+'.pk', 'rb'))
x_test_data_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx2+'.pk', 'rb'))
signal_pred_time_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx2+'.pk', 'rb'))
signal_pred_data_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx2+'.pk', 'rb'))
mae_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx2+'.pk', 'rb'))
mse_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx2+'.pk', 'rb'))
rmse_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx2+'.pk', 'rb'))
trac_02=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx2+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_02 = np.concatenate(signal_pred_time_02, axis=0, out=None)
signal_pred_data_all_02 = np.concatenate(signal_pred_data_02, axis=0, out=None) 
# time_org_all_01 = np.concatenate(time_org_01, axis=0, out=None)
# x_train_data_all_01 = np.concatenate(x_train_data_01, axis=0, out=None) 
x_test_data_all_02 = np.concatenate(x_test_data_02, axis=0, out=None)
mae_all_02 = (mae_02)
mse_all_02=(mse_02)
rmse_all_02 = (rmse_02)
trac_all_02=(trac_02)
#%% learn 0.03 s, pred 1s
saveNameIdx3='301000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx3+'.pk', 'rb'))
signal_pred_time_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx3+'.pk', 'rb'))
signal_pred_data_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx3+'.pk', 'rb'))
mae_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx3+'.pk', 'rb'))
mse_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx3+'.pk', 'rb'))
rmse_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx3+'.pk', 'rb'))
trac_03=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx3+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_03 = np.concatenate(signal_pred_time_03, axis=0, out=None)
signal_pred_data_all_03 = np.concatenate(signal_pred_data_03, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_03 = np.concatenate(x_test_data_03, axis=0, out=None)
mae_all_03 = (mae_03)
mse_all_03=(mse_03)
rmse_all_03 = (rmse_03)
trac_all_03=(trac_03)
#%% learn 0.04 s, pred 1s
saveNameIdx4='401000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx4+'.pk', 'rb'))
signal_pred_time_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx4+'.pk', 'rb'))
signal_pred_data_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx4+'.pk', 'rb'))
mae_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx4+'.pk', 'rb'))
mse_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx4+'.pk', 'rb'))
rmse_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx4+'.pk', 'rb'))
trac_04=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx4+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_04 = np.concatenate(signal_pred_time_04, axis=0, out=None)
signal_pred_data_all_04 = np.concatenate(signal_pred_data_04, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_04 = np.concatenate(x_test_data_04, axis=0, out=None)
mae_all_04 = (mae_04)
mse_all_04=(mse_04)
rmse_all_04 = (rmse_04)
trac_all_04=(trac_04)

#%% learn 0.05 s, pred 1s
saveNameIdx5='501000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx5+'.pk', 'rb'))
signal_pred_time_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx5+'.pk', 'rb'))
signal_pred_data_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx5+'.pk', 'rb'))
mae_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx5+'.pk', 'rb'))
mse_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx5+'.pk', 'rb'))
rmse_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx5+'.pk', 'rb'))
trac_05=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx5+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_05 = np.concatenate(signal_pred_time_05, axis=0, out=None)
signal_pred_data_all_05 = np.concatenate(signal_pred_data_05, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_05 = np.concatenate(x_test_data_05, axis=0, out=None)
mae_all_05 = (mae_05)
mse_all_05=(mse_05)
rmse_all_05 = (rmse_05)
trac_all_05=(trac_05)
#%% learn 0.06 s, pred 1s
saveNameIdx6='601000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx6+'.pk', 'rb'))
signal_pred_time_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx6+'.pk', 'rb'))
signal_pred_data_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx6+'.pk', 'rb'))
mae_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx6+'.pk', 'rb'))
mse_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx6+'.pk', 'rb'))
rmse_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx6+'.pk', 'rb'))
trac_06=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx6+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_06 = np.concatenate(signal_pred_time_06, axis=0, out=None)
signal_pred_data_all_06 = np.concatenate(signal_pred_data_06, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_06 = np.concatenate(x_test_data_06, axis=0, out=None)
mae_all_06 = (mae_06)
mse_all_06=(mse_06)
rmse_all_06 = (rmse_06)
trac_all_06=(trac_06)
#%% learn 0.07 s, pred 1s
saveNameIdx7='701000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx7+'.pk', 'rb'))
signal_pred_time_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx7+'.pk', 'rb'))
signal_pred_data_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx7+'.pk', 'rb'))
mae_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx7+'.pk', 'rb'))
mse_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx7+'.pk', 'rb'))
rmse_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx7+'.pk', 'rb'))
trac_07=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx7+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_07 = np.concatenate(signal_pred_time_07, axis=0, out=None)
signal_pred_data_all_07 = np.concatenate(signal_pred_data_07, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_07 = np.concatenate(x_test_data_07, axis=0, out=None)
mae_all_07 = (mae_07)
mse_all_07=(mse_07)
rmse_all_07 = (rmse_07)
trac_all_07=(trac_07)
#%% learn 0.08 s, pred 1s
saveNameIdx8='801000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx8+'.pk', 'rb'))
signal_pred_time_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx8+'.pk', 'rb'))
signal_pred_data_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx8+'.pk', 'rb'))
mae_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx8+'.pk', 'rb'))
mse_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx8+'.pk', 'rb'))
rmse_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx8+'.pk', 'rb'))
trac_08=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx8+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_08 = np.concatenate(signal_pred_time_08, axis=0, out=None)
signal_pred_data_all_08 = np.concatenate(signal_pred_data_08, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_08 = np.concatenate(x_test_data_08, axis=0, out=None)
mae_all_08 = (mae_08)
mse_all_08=(mse_08)
rmse_all_08 = (rmse_08)
trac_all_08=(trac_08)
#%% learn 0.09s, pred 1s
saveNameIdx9='901000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx9+'.pk', 'rb'))
signal_pred_time_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx9+'.pk', 'rb'))
signal_pred_data_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx9+'.pk', 'rb'))
mae_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx9+'.pk', 'rb'))
mse_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx9+'.pk', 'rb'))
rmse_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx9+'.pk', 'rb'))
trac_09=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx9+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_09 = np.concatenate(signal_pred_time_09, axis=0, out=None)
signal_pred_data_all_09 = np.concatenate(signal_pred_data_09, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_09 = np.concatenate(x_test_data_09, axis=0, out=None)
mae_all_09 = (mae_09)
mse_all_09=(mse_09)
rmse_all_09 = (rmse_09)
trac_all_09=(trac_09)
#%% learn 0.1 s, pred 1s
saveNameIdx10='1001000100' 
# time_org_001=pickle.load(open('time_org'+saveNameIdx3+'.pk','rb'))
# x_train_data_001=pickle.load (open('x_train_data'+saveNameIdx3+'.pk', 'rb'))
x_test_data_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx10+'.pk', 'rb'))
signal_pred_time_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx10+'.pk', 'rb'))
signal_pred_data_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx10+'.pk', 'rb'))
mae_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx10+'.pk', 'rb'))
mse_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx10+'.pk', 'rb'))
rmse_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx10+'.pk', 'rb'))
trac_p1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx10+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p1 = np.concatenate(signal_pred_time_p1, axis=0, out=None)
signal_pred_data_all_p1 = np.concatenate(signal_pred_data_p1, axis=0, out=None) 
# time_org_all_001 = np.concatenate(time_org_001, axis=0, out=None)
# x_train_data_all_001 = np.concatenate(x_train_data_001, axis=0, out=None) 
x_test_data_all_p1 = np.concatenate(x_test_data_p1, axis=0, out=None)
mae_all_p1 = (mae_p1)
mse_all_p1=(mse_p1)
rmse_all_p1 = (rmse_p1)
trac_all_p1=(trac_p1)
#%% learn 1 s, pred 1s
saveNameIdx11='10001000100' 
# time_org_1=pickle.load(open('time_org'+saveNameIdx1+'.pk','rb'))
# x_train_data_1=pickle.load (open('x_train_data'+saveNameIdx1+'.pk', 'rb'))
x_test_data_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx11+'.pk', 'rb'))
signal_pred_time_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx11+'.pk', 'rb'))
signal_pred_data_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx11+'.pk', 'rb'))
mae_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx11+'.pk', 'rb'))
mse_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx11+'.pk', 'rb'))
rmse_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx11+'.pk', 'rb'))
trac_1=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx11+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_1 = np.concatenate(signal_pred_time_1, axis=0, out=None)
signal_pred_data_all_1 = np.concatenate(signal_pred_data_1, axis=0, out=None) 
# time_org_all_1 = np.concatenate(time_org_1, axis=0, out=None)
# x_train_data_all_1 = np.concatenate(x_train_data_1, axis=0, out=None) 
x_test_data_all_1 = np.concatenate(x_test_data_1, axis=0, out=None)
mae_all_1 = (mae_1)
mse_all_1=(mse_1)
rmse_all_1 = (rmse_1)
trac_all_1=(trac_1)

#%% learn .5 s, pred 1s
saveNameIdx12='5001000100' 
# time_org_01=pickle.load(open('time_org'+saveNameIdx2+'.pk','rb'))
# x_train_data_01=pickle.load (open('x_train_data'+saveNameIdx2+'.pk', 'rb'))
x_test_data_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx12+'.pk', 'rb'))
signal_pred_time_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx12+'.pk', 'rb'))
signal_pred_data_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx12+'.pk', 'rb'))
mae_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx12+'.pk', 'rb'))
mse_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx12+'.pk', 'rb'))
rmse_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx12+'.pk', 'rb'))
trac_p5=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx12+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p5 = np.concatenate(signal_pred_time_p5, axis=0, out=None)
signal_pred_data_all_p5 = np.concatenate(signal_pred_data_p5, axis=0, out=None) 
# time_org_all_01 = np.concatenate(time_org_01, axis=0, out=None)
# x_train_data_all_01 = np.concatenate(x_train_data_01, axis=0, out=None) 
x_test_data_all_p5 = np.concatenate(x_test_data_p5, axis=0, out=None)
mae_all_p5 = (mae_p5)
mse_all_p5=(mse_p5)
rmse_all_p5 = (rmse_p5)
trac_all_p5=(trac_p5)



#%% plotting with specific parameters
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
   'figure.figsize': [6.5,6.5] # width and height in inch (max width = 7.5", max length = 10")
   }
plt.rcParams.update(params)
# #%% predic3tion and valid/testing plot

fig1 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=6, figure=fig1)
ax = fig1.add_subplot(spec2[0, 0])
ax.plot(Sig_pred_time_all_01,x_test_data_all_01,'-',linewidth=.5, label='data')
ax.plot(Sig_pred_time_all_01,signal_pred_data_all_01,'--',linewidth=.5, label='predicted')
# plt.xlabel('time (s)\n (a)')
# ax.set_title("Hello Title", y=0, pad=-35, verticalalignment="bottom")
ax.set_title('(a)', y=-.5, ha='center')
# plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
# plt.title('(a)', y=-.4, ha='center')
plt.xlim(0,1.5)
plt.grid()
plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

ax = fig1.add_subplot(spec2[0, 1])
ax.plot(Sig_pred_time_all_02,x_test_data_all_02,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_02,signal_pred_data_all_02,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(b)', y=-.5, ha='center')
plt.xlim(0,1.5)
plt.grid()
# plt.legend()

ax = fig1.add_subplot(spec2[1, 0])
ax.plot(Sig_pred_time_all_03,x_test_data_all_03,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_03,signal_pred_data_all_03,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
plt.title('(c)', y=-.5, ha='center')
plt.xlim(0,1.5)
plt.grid()
fig1.tight_layout()
# plt.legend()

ax = fig1.add_subplot(spec2[1, 1])
ax.plot(Sig_pred_time_all_04,x_test_data_all_04,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_04,signal_pred_data_all_04,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(d)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[2, 0])
ax.plot(Sig_pred_time_all_05,x_test_data_all_05,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_05,signal_pred_data_all_05,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
plt.title('(e)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[2, 1])
ax.plot(Sig_pred_time_all_06,x_test_data_all_06,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_06,signal_pred_data_all_06,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(f)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[3, 0])
ax.plot(Sig_pred_time_all_07,x_test_data_all_07,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_07,signal_pred_data_all_07,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
plt.title('(g)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[3, 1])
ax.plot(Sig_pred_time_all_08,x_test_data_all_08,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_08,signal_pred_data_all_08,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(h)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[4, 0])
ax.plot(Sig_pred_time_all_09,x_test_data_all_09,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_09,signal_pred_data_all_09,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
plt.title('(i)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[4, 1])
ax.plot(Sig_pred_time_all_p1,x_test_data_all_p1,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_p1,signal_pred_data_all_p1,'--',linewidth=.5, label='prediction')
# plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(j)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()

ax = fig1.add_subplot(spec2[5, 0])
ax.plot(Sig_pred_time_all_p5,x_test_data_all_p5,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_p5,signal_pred_data_all_p5,'--',linewidth=.5, label='prediction')
plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s$^2$)')
# plt.suptitle('(k)', ha='center')
plt.title('(k)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()


ax = fig1.add_subplot(spec2[5, 1])
ax.plot(Sig_pred_time_all_1,x_test_data_all_1,'-',linewidth=.5, label='valid')
ax.plot(Sig_pred_time_all_1,signal_pred_data_all_1,'--',linewidth=.5, label='prediction')
plt.xlabel('time (s)')
# plt.ylabel('acceleration (m/$s^2$)')
plt.title('(l)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig1.tight_layout()
plt.savefig('plots/train_predic_zoom_all_shortest_learning_100ms_CT', dpi=400)
plt.savefig('plots/train_predic_zoom_all_shortest_learning_100ms_CT.pdf', dpi=400)



#%% error
fig2 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=6, figure=fig2)
ax = fig2.add_subplot(spec2[0, 0])
ax.plot(Sig_pred_time_all_01,abs(x_test_data_all_01-signal_pred_data_all_01),'-',linewidth=.5, label='error')#label= 'error for 0.01 sec learning window'
# plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(a)', y=-.5, ha='center')
plt.xlim(0,1.5)
plt.grid()
# plt.legend()
plt.legend(loc=2,ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)
ax = fig2.add_subplot(spec2[0, 1])
ax.plot(Sig_pred_time_all_02,abs(x_test_data_all_02-signal_pred_data_all_02),'-',linewidth=.5, label='error for 0.02 sec learning window')
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(b)', y=-.5, ha='center')
plt.xlim(0,1.5)
plt.grid()
# plt.legend()

ax = fig2.add_subplot(spec2[1, 0])
ax.plot(Sig_pred_time_all_03,abs(x_test_data_all_03-signal_pred_data_all_03),'-',linewidth=.5, label='error for 0.03 sec learning window')
# plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(c)', y=-.5, ha='center')
plt.xlim(0,1.5)
plt.grid()
fig2.tight_layout()
# plt.legend()

ax = fig2.add_subplot(spec2[1, 1])
ax.plot(Sig_pred_time_all_04,abs(x_test_data_all_04-signal_pred_data_all_04),'-',linewidth=.5, label='error for 0.04 sec learning window')
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(d)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[2, 0])
ax.plot(Sig_pred_time_all_05,abs(x_test_data_all_05-signal_pred_data_all_05),'-',linewidth=.5, label='error for 0.05 sec learning window')
# plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(e)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[2, 1])
ax.plot(Sig_pred_time_all_06,abs(x_test_data_all_06-signal_pred_data_all_06),'-',linewidth=.5, label='error for 0.06 sec learning window')
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(f)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[3, 0])
ax.plot(Sig_pred_time_all_07,abs(x_test_data_all_07-signal_pred_data_all_07),'-',linewidth=.5, label='error for 0.07 sec learning window')
# plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(g)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[3, 1])
ax.plot(Sig_pred_time_all_08,abs(x_test_data_all_08-signal_pred_data_all_08),'-',linewidth=.5, label='error for 0.08 sec learning window')
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(h)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[4, 0])
ax.plot(Sig_pred_time_all_09,abs(x_test_data_all_09-signal_pred_data_all_09),'-',linewidth=.5, label='error for 0.09 sec learning window')
# plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(i)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[4, 1])
ax.plot(Sig_pred_time_all_p1,abs(x_test_data_all_p1-signal_pred_data_all_p1),'-',linewidth=.5, label='error for 0.1 sec learning window')
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(j)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()

ax = fig2.add_subplot(spec2[5, 0])
ax.plot(Sig_pred_time_all_p5,abs(x_test_data_all_p5-signal_pred_data_all_p5),'-',linewidth=.5, label='error for 0.5 sec learning window')
plt.xlabel('time (s)')
plt.ylabel('error (m/s$^2$)')
plt.title('(k)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()


ax = fig2.add_subplot(spec2[5, 1])
ax.plot(Sig_pred_time_all_1,abs(x_test_data_all_1-signal_pred_data_all_1),'-',linewidth=.5, label='error for 1 sec learning window')
plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
plt.title('(l)', y=-.5, ha='center')
plt.xlim(0,1.5)
# plt.legend()
plt.grid()
fig2.tight_layout()


plt.savefig('plots/error _all_shortest_learning_100ms_CT', dpi=400)
plt.savefig('plots/error _all_shortest_learning_100ms_CT.pdf', dpi=400)
