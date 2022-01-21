# -*- coding: utf-8 -*-


"""
Created on Thu Apr  1 16:59:19 2021

@author: chypuja

  Effect of various computational time (T) in a specific
learning window length (L) showing: (a) MAE in different states, and;
(b) transient time  (SAMASOIS paper figure 8)
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


#%% learn 0.1 s, pred 1s
saveNameIdx10='1001000100' 
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
x_test_data_all_p1 = np.concatenate(x_test_data_p1, axis=0, out=None)
mae_all_p1 = (mae_p1)
mse_all_p1=(mse_p1)
rmse_all_p1 = (rmse_p1)
trac_all_p1=(trac_p1)
#%% learn 1 s, pred 1s
saveNameIdx11='10001000100' 
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
x_test_data_all_1 = np.concatenate(x_test_data_1, axis=0, out=None)
mae_all_1 = (mae_1)
mse_all_1=(mse_1)
rmse_all_1 = (rmse_1)
trac_all_1=(trac_1)

#%% learn .1 s, pred 1s CT=100ms/.1
saveNameIdx12='1001000100' 
x_test_data_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx12+'.pk', 'rb'))
signal_pred_time_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx12+'.pk', 'rb'))
signal_pred_data_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx12+'.pk', 'rb'))
mae_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx12+'.pk', 'rb'))
mse_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx12+'.pk', 'rb'))
rmse_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx12+'.pk', 'rb'))
trac_p1_ct100=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx12+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p1_ct100 = np.concatenate(signal_pred_time_p1_ct100, axis=0, out=None)
signal_pred_data_all_p1_ct100 = np.concatenate(signal_pred_data_p1_ct100, axis=0, out=None) 
x_test_data_all_p1_ct100 = np.concatenate(x_test_data_p1_ct100, axis=0, out=None)
mae_all_p1_ct100 = (mae_p1_ct100)
mse_all_p1_ct100=(mse_p1_ct100)
rmse_all_p1_ct100 = (rmse_p1_ct100)
trac_all_p1_ct100=(trac_p1_ct100)
#%% learn .1 s, pred 1s, CT=10ms/.01
saveNameIdx13='100100010' 
x_test_data_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx13+'.pk', 'rb'))
signal_pred_time_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx13+'.pk', 'rb'))
signal_pred_data_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx13+'.pk', 'rb'))
mae_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx13+'.pk', 'rb'))
mse_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx13+'.pk', 'rb'))
rmse_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx13+'.pk', 'rb'))
trac_p1_ct10=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx13+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p1_ct10 = np.concatenate(signal_pred_time_p1_ct10, axis=0, out=None)
signal_pred_data_all_p1_ct10 = np.concatenate(signal_pred_data_p1_ct10, axis=0, out=None) 
x_test_data_all_p1_ct10 = np.concatenate(x_test_data_p1_ct10, axis=0, out=None)
mae_all_p1_ct10 = (mae_p1_ct10)
mse_all_p1_ct10=(mse_p1_ct10)
rmse_all_p1_ct10 = (rmse_p1_ct10)
trac_all_p1_ct10=(trac_p1_ct10)
#%% learn .1 s, pred 1s, CT=1000ms/1s
saveNameIdx14='10010001000' 
x_test_data_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx14+'.pk', 'rb'))
signal_pred_time_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx14+'.pk', 'rb'))
signal_pred_data_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx14+'.pk', 'rb'))
mae_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx14+'.pk', 'rb'))
mse_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx14+'.pk', 'rb'))
rmse_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx14+'.pk', 'rb'))
trac_p1_ct1s=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx14+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p1_ct1s = np.concatenate(signal_pred_time_p1_ct1s, axis=0, out=None)
signal_pred_data_all_p1_ct1s = np.concatenate(signal_pred_data_p1_ct1s, axis=0, out=None) 
x_test_data_all_p1_ct1s = np.concatenate(x_test_data_p1_ct1s, axis=0, out=None)
mae_all_p1_ct1s = (mae_p1_ct1s)
mse_all_p1_ct1s=(mse_p1_ct1s)
rmse_all_p1_ct1s = (rmse_p1_ct1s)
trac_all_p1_ct1s=(trac_p1_ct1s)
#%% learn .1 s, pred 1s, CT=500ms/.5s
saveNameIdx15='1001000500' 
x_test_data_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/x_test_data'+saveNameIdx15+'.pk', 'rb'))
signal_pred_time_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_time'+saveNameIdx15+'.pk', 'rb'))
signal_pred_data_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/signal_pred_data'+saveNameIdx15+'.pk', 'rb'))
mae_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mae'+saveNameIdx15+'.pk', 'rb'))
mse_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/mse'+saveNameIdx15+'.pk', 'rb'))
rmse_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/rmse'+saveNameIdx15+'.pk', 'rb'))
trac_p1_ct500=pickle.load (open('pickel_1sec_.5sec_.1sec_learning_for_1s_pred_slide/trac'+saveNameIdx15+'.pk', 'rb'))

#%% Merging data
Sig_pred_time_all_p1_ct500 = np.concatenate(signal_pred_time_p1_ct500, axis=0, out=None)
signal_pred_data_all_p1_ct500 = np.concatenate(signal_pred_data_p1_ct500, axis=0, out=None) 
x_test_data_all_p1_ct500 = np.concatenate(x_test_data_p1_ct500, axis=0, out=None)
mae_all_p1_ct500 = (mae_p1_ct500)
mse_all_p1_ct500=(mse_p1_ct500)
rmse_all_p1_ct500 = (rmse_p1_ct500)
trac_all_p1_ct500=(trac_p1_ct500)
#%% plotting with specific parameters
# Reseting plot parameter to default
plt.rcdefaults()
# Updating Parameters for Paper
params = {
    'lines.linewidth' :1,
    'lines.markersize' :4,
   'axes.labelsize': 8,
    'axes.titlesize':8,
    'axes.titleweight':'normal',
    'font.size': 8,
    'font.family': 'Times New Roman', # 'Times New RomanArial'
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'legend.shadow':'False',
   'legend.fontsize': 6,
   'xtick.labelsize': 6,
   'ytick.labelsize': 6,
   'text.usetex': False,
    'figure.autolayout': True,
   'figure.figsize': [2.5,4] # width and height in inch (max width = 7.5", max length = 10")
   }
plt.rcParams.update(params)


#%%
def dataForTrans(data, data_time, tr_du):
    strt_trans = np.where(data_time==0)[0][0]
    trans_samples = int(tr_du*51200)
    stop_trans = strt_trans+trans_samples
    mean_preTrans = np.mean(data[:strt_trans])
    mean_Trans = np.mean(data[strt_trans:stop_trans])
    mean_postTrans = np.mean(data[stop_trans:])
    conv = [mean_preTrans,mean_Trans,mean_postTrans]
    # convTime = [data_time[np.where(data_time==-4)[0][0]],data_time[np.where(data_time==0)[0][0]],data_time[np.where(data_time==+4)[0][0]]]
    return conv 

error_all_p1_ct100 = abs(x_test_data_all_p1_ct100-signal_pred_data_all_p1_ct100)
error_all_p1_ct10=abs(x_test_data_all_p1_ct10-signal_pred_data_all_p1_ct10)
error_all_p1_ct1s = abs(x_test_data_all_p1_ct1s-signal_pred_data_all_p1_ct1s)
error_all_p1_ct500 = abs(x_test_data_all_p1_ct500-signal_pred_data_all_p1_ct500)

conv_p1_ct100 = dataForTrans(error_all_p1_ct100,Sig_pred_time_all_p1_ct100,0.42)    
conv_p1_ct10 = dataForTrans(error_all_p1_ct10,Sig_pred_time_all_p1_ct10, 0.32)   
conv_p1_ct1s = dataForTrans(error_all_p1_ct1s,Sig_pred_time_all_p1_ct1s, 1.32) 
conv_p1_ct500 = dataForTrans(error_all_p1_ct500,Sig_pred_time_all_p1_ct500, 0.82)
convTime = [2, 5, 8]
convTimetick = ['pre-event steady \n state', 'transient event', 'post-event steady \n state']  




#%% two figure togather
# mean error
fig1 = plt.figure()
ax = fig1.add_subplot(211)
ax.plot(convTime,conv_p1_ct10,marker='^',ls='solid',label=' L= 0.1 s, T= 0.01 s') # L= learning window length, T= computational time
ax.plot(convTime,conv_p1_ct100,marker='o',ls='dashed',label='L= 0.1 s, T= 0.1 s')
ax.plot(convTime,conv_p1_ct500,marker='s',ls='dashdot',label='L= 0.1 s, T= 0.5 s')
ax.plot(convTime,conv_p1_ct1s,marker='+',ls='dotted',label='L= 0.1 s, T= 1.0 s')

ax.plot([3.5,3.5],[-.001,.069],'--',linewidth=1,color='black')
ax.plot([6.5,6.5],[-.001,.069],'--',linewidth=1,color='black')
ax.yaxis.grid()
ax.set_axisbelow(True)
plt.xticks(convTime,convTimetick)
plt.ylabel('MAE (m/s$^2$)')
plt.title('(a)', y=-.32, ha='center') # y=-.29
fig1.tight_layout()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.27),ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

# transient time
mean1 =[0.32] #  LT=0.1s, CT=10 ms/0.01 s transition time =0.325, find from original error plot
mean2 =[0.42] #  LT=0.1s, CT=100 ms/0.1 s transition time =0.425, find from original error plot
mean3 = [1.32] #  LT=0.1s, CT=1000 ms/01 s transition time =1.325, find from original error plot
mean4 = [0.82] #  LT=0.1s, CT=500 ms/.5 s transition time =0.72, find from original error plot

ax = fig1.add_subplot(212)
exp_set = ['0.01 s', '0.1 s','0.5 s', '1.0 s']
x = np.arange(len(exp_set)) 
mean = [0.32,0.42,0.82,1.32]
hatch=['---','///','...','\\\\\\']
label=[' L= 0.1 s, T= 0.01 s','L= 0.1 s, T= 0.1 s','L= 0.1 s, T= 0.5 s',' L= 0.1 s, T= 1.0 s']
ax.bar(x[0],mean[0], hatch=hatch[0],label=label[0])
ax.bar(x[1],mean[1],hatch=hatch[1],label=label[1])
ax.bar(x[2],mean[2],hatch=hatch[2],label=label[2])
ax.bar(x[3],mean[3],hatch=hatch[3],label=label[3])

ax.set_ylabel('transient time (s)')
plt.xticks(x,exp_set)
plt.title('(b)', y=-.32, ha='center')
fig1.tight_layout()
plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.27),ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)
plt.show()
plt.savefig('plots/MAE_transition_time_4various_T_p1_L.pdf', dpi=400)
plt.savefig('plots/MAE_transition_time_4various_T_p1_L.png', dpi=400)
























































