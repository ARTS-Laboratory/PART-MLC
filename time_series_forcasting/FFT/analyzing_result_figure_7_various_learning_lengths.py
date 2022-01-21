


"""
Created on Sun Mar 14 21:38:40 2021

@author: chypuja

plot Effect of various learning window lengths (L) showing:
(a) MAE in different states, and; (b) transient time.(SAMASIS paper figure 7)
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


#%% learn 0.1 s, pred 1s, CT=100ms/0.1 s
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
#%% learn 1 s, pred 1s, CT=10ms/0.01 s
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

#%% learn .5 s, pred 1s, CT=1ms/0.001 s
saveNameIdx12='5001000100' 
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
   'xtick.labelsize':6,
   'ytick.labelsize':6,
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
    return conv 

error_all_p1 = abs(x_test_data_all_p1-signal_pred_data_all_p1)
error_all_p5=abs(x_test_data_all_p5-signal_pred_data_all_p5)
error_all_1 = abs(x_test_data_all_1-signal_pred_data_all_1)

conv_p1 = dataForTrans(error_all_p1,Sig_pred_time_all_p1, 0.42)    
conv_p5 = dataForTrans(error_all_p5,Sig_pred_time_all_p5, 0.82)   
conv_1 = dataForTrans(error_all_1,Sig_pred_time_all_1, 1.32) 
convTime = [2, 5, 8]
convTimetick = ['pre-event steady\n state', 'transient event', 'post-event steady\n state']  # \n for enter second row

#%% two figure togather
# mean error
fig1 = plt.figure()
ax = fig1.add_subplot(211)

ax.plot(convTime,conv_p1,marker='o',ls='solid',label='L= 0.1 s')
ax.plot(convTime,conv_p5,marker='^',ls='dashed',label=' L= 0.5 s')
ax.plot(convTime,conv_1,marker='+',ls='dotted',label=' L= 1.0 s')
ax.plot([3.5,3.5],[-.001,.045],'--',linewidth=1,color='black')
ax.plot([6.5,6.5],[-.001,.045],'--',linewidth=1,color='black')
ax.yaxis.grid()
ax.set_axisbelow(True)
plt.xticks(convTime,convTimetick)
plt.ylabel('MAE (m/s$^2$)')
plt.title('(a)', y=-.29, ha='center')
fig1.tight_layout()
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.16),ncol=3,facecolor='white', edgecolor = 'black', framealpha=1)

# transient time
mean1=0.42  # L=0.1 s
mean2=0.82 # L= 0.5 s
mean3=1.32 # L=1 s
ax = fig1.add_subplot(212)
exp_set = ['0.1 s', '0.5 s', '1.0 s']
x = np.arange(len(exp_set)) 
mean = [0.4246,0.825,1.325]

hatch=['---','///','...']
label=['L= 0.1 s',' L= 0.5 s',' L= 1.0 s']
ax.bar(x[0],mean[0], hatch=hatch[0],label=label[0])
ax.bar(x[1],mean[1],hatch=hatch[1],label=label[1])
ax.bar(x[2],mean[2],hatch=hatch[2],label=label[2])

ax.set_ylabel('transient time (s)')
# ax.set_xlabel('learning length (s)')

plt.xticks(x,exp_set)
plt.title('(b)', y=-.29, ha='center')
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.16),ncol=3,facecolor='white', edgecolor = 'black', framealpha=1)
fig1.tight_layout()
plt.show()

plt.savefig('plots/MAE_transition_time_various_L.pdf', dpi=400)
plt.savefig('plots/MAE_transition_time_various_L.png', dpi=400)



































































# #%%
# #%% Data post-processing for plot .1 seconds learning
# import sklearn
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

# # adjusting by deleting 0.1 seconds from the first and from the last of the signals
# x_test_data_all_p1_adjust = x_test_data_all_p1
# signal_pred_data_all_p1_adjust = signal_pred_data_all_p1
# Sig_pred_time_all_p1_adjust = Sig_pred_time_all_p1

# # reshaping for plot
# signal_pred_time_p1_pp = np.reshape(Sig_pred_time_all_p1_adjust[0:(-int((1-post_sliding)*51200))],(-1,int(post_sliding*51200))) # -1 mean row size doesn't defined

# # adjusted error
# abs_error_adjusted_p1 = abs(x_test_data_all_p1_adjust-signal_pred_data_all_p1_adjust)

# # calculaing mean abs error again based on the new shape
# mae_all_p1_pp=list(np.mean(np.reshape(abs_error_adjusted_p1[0:(-int((1-post_sliding)*51200))],(-1,int(post_sliding*51200))),axis=1)) # minus, reshape, mean

# # calculaing mse again based on the new shape
# # mse_all_p1_pp=list(np.mean(np.reshape(abs_error_adjusted_p1**2,(-1,int(0.4*51200))),axis=1))

# # calculaing rmse again based on the new shape
# rmse_all_p1_pp=list(np.sqrt(np.mean(np.reshape(abs_error_adjusted_p1[0:(-int((1-post_sliding)*51200))]**2,(-1,int(post_sliding*51200))),axis=1)))

# # calculating trac based on the new shape
# # x_test_data_01_pp = np.reshape(x_test_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # signal_pred_data_01_pp = np.reshape(signal_pred_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # trac_all_1_pp=[]
# # for i in range(0,x_test_data_1_pp.shape[0]):
# #     trac_all_1_pp.append((np.dot(x_test_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])** 2/
# #                           ((np.dot(x_test_data_1_pp[i,:].conj().T, x_test_data_1_pp[i,:])) *
# #            
# # #%% Data post-processing for plot .01 seconds learning as there is 0.09 second sliding missmatch with .1seconds learning 
# # import sklearn
# # from sklearn.metrics import mean_squared_error
# # from sklearn.metrics import mean_absolute_error

# # # adjusting by deleting 0.1 seconds from the first and from the last of the signals
# # x_test_data_all_01_adjust = x_test_data_all_01[int(0.09*51200):-1*int(0.91*51200)]
# # signal_pred_data_all_01_adjust = signal_pred_data_all_01[int(0.09*51200):-1*int(0.91*51200)]
# # Sig_pred_time_all_01_adjust = Sig_pred_time_all_01[int(0.09*51200):-1*int(0.91*51200)]

# # # reshaping for plot
# # # signal_pred_time_01_pp = np.reshape(Sig_pred_time_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # signal_pred_time_01_pp = np.reshape(Sig_pred_time_all_01_adjust[0:(-int((1-post_sliding)*51200))],(-1,int(post_sliding*51200)))
# # # adjusted error
# # abs_error_adjusted_01 = abs(x_test_data_all_01_adjust-signal_pred_data_all_01_adjust)

# # # calculaing mean abs error again based on the new shape
# # mae_all_01_pp=list(np.mean(np.reshape(abs_error_adjusted_01[0:(-int((1-post_sliding)*51200))],(-1,int(post_sliding*51200))),axis=1)) # minus, reshape, mean

# # # calculaing mse again based on the new shape
# # # mse_all_01_pp=list(np.mean(np.reshape(abs_error_adjusted_01**2,(-1,int(1*51200))),axis=1))

# # # calculaing rmse again based on the new shape
# # # rmse_all_01_pp=list(np.sqrt(np.mean(np.reshape(abs_error_adjusted_01**2,(-1,int(1*51200))),axis=1)))

# # # calculating trac based on the new shape
# # # x_test_data_01_pp = np.reshape(x_test_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # # signal_pred_data_01_pp = np.reshape(signal_pred_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # # trac_all_1_pp=[]
# # # for i in range(0,x_test_data_1_pp.shape[0]):
# # #     trac_all_1_pp.append((np.dot(x_test_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])** 2/
# # #                           ((np.dot(x_test_data_1_pp[i,:].conj().T, x_test_data_1_pp[i,:])) *
# # #                            (np.dot(signal_pred_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])))))    
# # #%% Data post-processing for plot .25 seconds learning as there is 0.15 second sliding missmatch with .1seconds learning 
# # import sklearn
# # from sklearn.metrics import mean_squared_error
# # from sklearn.metrics import mean_absolute_error

# # # adjusting by deleting 0.1 seconds from the first and from the last of the signals
# # x_test_data_all_p25_adjust = x_test_data_all_p25[int(0.85*51200):-1*int(0.15*51200)]
# # signal_pred_data_all_p25_adjust = signal_pred_data_all_p25[int(0.85*51200):-1*int(0.15*51200)]
# # Sig_pred_time_all_p25_adjust = Sig_pred_time_all_p25[int(0.85*51200):-1*int(0.15*51200)]

# # # reshaping for plot
# # signal_pred_time_p25_pp = np.reshape(Sig_pred_time_all_p25_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # # adjusted error
# # abs_error_adjusted_p25 = abs(x_test_data_all_p25_adjust-signal_pred_data_all_p25_adjust)

# # # calculaing mean abs error again based on the new shape
# # mae_all_p25_pp=list(np.mean(np.reshape(abs_error_adjusted_p25,(-1,int(1*51200))),axis=1)) # minus, reshape, mean

# # # calculaing mse again based on the new shape
# # mse_all_p25_pp=list(np.mean(np.reshape(abs_error_adjusted_p25**2,(-1,int(1*51200))),axis=1))

# # # calculaing rmse again based on the new shape
# # rmse_all_p25_pp=list(np.sqrt(np.mean(np.reshape(abs_error_adjusted_p25[0:(-int((1-post_sliding)*51200))]**2,(-1,int(0.4*51200))),axis=1)))

# # # calculating trac based on the new shape
# # # x_test_data_01_pp = np.reshape(x_test_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # # signal_pred_data_01_pp = np.reshape(signal_pred_data_all_01_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # # trac_all_1_pp=[]
# # # for i in range(0,x_test_data_1_pp.shape[0]):
# # #     trac_all_1_pp.append((np.dot(x_test_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])** 2/
# # #                           ((np.dot(x_test_data_1_pp[i,:].conj().T, x_test_data_1_pp[i,:])) *
# # #                            (np.dot(signal_pred_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:]))))) 

# #%% Data post-processing for plot 1 seconds learning as there is 0.1 second sliding missmatch
# import sklearn
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

# # adjusting by deleting 0.1 seconds from the first and from the last of the signals
# x_test_data_all_1_adjust = x_test_data_all_1[int(0.1*51200):-1*int(0.9*51200)]
# signal_pred_data_all_1_adjust = signal_pred_data_all_1[int(0.1*51200):-1*int(0.9*51200)]
# Sig_pred_time_all_1_adjust = Sig_pred_time_all_1[int(0.1*51200):-1*int(0.9*51200)]

# # reshaping for plot
# signal_pred_time_1_pp = np.reshape(Sig_pred_time_all_1_adjust,(-1,int(post_sliding*51200))) # -1 mean row size doesn't defined

# # adjusted error
# abs_error_adjusted_1 = abs(x_test_data_all_1_adjust-signal_pred_data_all_1_adjust)

# # calculaing mean abs error again based on the new shape
# mae_all_1_pp=list(np.mean(np.reshape(abs_error_adjusted_1,(-1,int(post_sliding*51200))),axis=1)) # minus, reshape, mean

# # calculaing mse again based on the new shape
# # mse_all_1_pp=list(np.mean(np.reshape(abs_error_adjusted_1**2,(-1,int(1*51200))),axis=1))

# # calculaing rmse again based on the new shape
# rmse_all_1_pp=list(np.sqrt(np.mean(np.reshape(abs_error_adjusted_1**2,(-1,int(post_sliding*51200))),axis=1)))

# # calculating trac based on the new shape
# # x_test_data_1_pp = np.reshape(x_test_data_all_1_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # signal_pred_data_1_pp = np.reshape(signal_pred_data_all_1_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # trac_all_1_pp=[]
# # for i in range(0,x_test_data_1_pp.shape[0]):
# #     trac_all_1_pp.append((np.dot(x_test_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])** 2/
# #                           ((np.dot(x_test_data_1_pp[i,:].conj().T, x_test_data_1_pp[i,:])) *
# #                             (np.dot(signal_pred_data_1_pp[i,:].conj().T, signal_pred_data_1_pp[i,:])))))    

# #%% Data post-processing for plot .5 seconds learning as there is 0.4 second sliding missmatch with .1 sec learning
# import sklearn
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
    
# # adjusting by deleting 0.4 seconds from the first and from the last of the signals
# x_test_data_all_p5_adjust = x_test_data_all_p5[int(0.6*51200):-1*int(0.4*51200)]
# signal_pred_data_all_p5_adjust = signal_pred_data_all_p5[int(0.6*51200):-1*int(0.4*51200)]
# Sig_pred_time_all_p5_adjust = Sig_pred_time_all_p5[int(0.6*51200):-1*int(0.4*51200)]

# # reshaping for plot
# signal_pred_time_p5_pp = np.reshape(Sig_pred_time_all_p5_adjust,(-1,int(post_sliding*51200))) 

# # adjusted error
# abs_error_adjusted_p5 = abs(x_test_data_all_p5_adjust-signal_pred_data_all_p5_adjust)

# # calculaing mean abs error again based on the new shape
# mae_all_p5_pp=list(np.mean(np.reshape(abs_error_adjusted_p5,(-1,int(post_sliding*51200))),axis=1)) # minus, reshape, mean

# # calculaing mse again based on the new shape
# # mse_all_p5_pp=list(np.mean(np.reshape(abs_error_adjusted_p5**2,(-1,int(1*51200))),axis=1))

# # calculaing rmse again based on the new shape
# rmse_all_p5_pp=list(np.sqrt(np.mean(np.reshape(abs_error_adjusted_p5**2,(-1,int(post_sliding*51200))),axis=1)))

# # # calculating trac based on the new shape
# # x_test_data_p5_pp = np.reshape(x_test_data_all_p5_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined
# # signal_pred_data_p5_pp = np.reshape(signal_pred_data_all_p5_adjust,(-1,int(1*51200))) # -1 mean row size doesn't defined

# # trac_all_01_pp=[]
# # for i in range(0,x_test_data_1_pp.shape[0]):
# #     trac_all_01_pp.append((np.dot(x_test_data_01_pp[i,:].conj().T, signal_pred_data_01_pp[i,:])** 2/
# #                           ((np.dot(x_test_data_01_pp[i,:].conj().T, x_test_data_01_pp[i,:])) *
# #                            (np.dot(signal_pred_data_01_pp[i,:].conj().T, signal_pred_data_01_pp[i,:]))))) 

# #%% mae/ rmse

# fig3 = plt.figure(figsize=(6.5,3))
# ax = fig3.add_subplot(211)
# # width = 1
# # ax.scatter(signal_pred_time_01_pp[:,0],mae_all_01_pp,marker='1',label=' 0.01 s learning length')
# ax.scatter((signal_pred_time_p1_pp)[:,0],mae_all_p1_pp,marker='o',label='0.1 s learning length')
# # ax.scatter((signal_pred_time_p25_pp)[:,0],mae_all_p25_pp,marker='D',label=' 0.25 s learning length')
# ax.scatter((signal_pred_time_p5_pp)[:,0],mae_all_p5_pp,marker='^',label=' 0.5 s learning length')
# ax.scatter((signal_pred_time_1_pp)[:,0],mae_all_1_pp,marker='+',label=' 1 s learning length')
# ax.plot([0,0],[-.001,3.2],'--',linewidth=1,color='black')
# ax.yaxis.grid()
# ax.set_axisbelow(True)
# # ax.set_yscale('log')
# # plt.semilogy(basey=2)
# # plt.xlim(-1,0)
# plt.xlabel('time (s)')
# plt.ylabel('MAE (m/$s^2$)')
# plt.title('(a)', y=-.7, ha='center')
# # plt.xlim(0,1)
# # plt.grid() 
# plt.legend(loc=2,bbox_to_anchor=(0, -0.55),facecolor='white',ncol=5, edgecolor = 'black', framealpha=1) #Using matplotlib.pyplot, plt.legend(facecolor='white', framealpha=1) will give your legend a white background without transparency.

# ax = fig3.add_subplot(212)
# # ax.scatter(signal_pred_time_01_pp[:,0],mae_all_01_pp,marker='1',label=' 0.01 s learning length')
# ax.scatter((signal_pred_time_p1_pp)[:,0],mae_all_p1_pp,marker='o',label='0.1 s learning length')
# # ax.scatter((signal_pred_time_p25_pp)[:,0],mae_all_p25_pp,marker='D',label=' 0.25 s learning length')
# ax.scatter((signal_pred_time_p5_pp)[:,0],mae_all_p5_pp,marker='^',label=' 0.5 s learning length')
# ax.scatter((signal_pred_time_1_pp)[:,0],mae_all_1_pp,marker='+',label=' 1 s learning length')
# ax.plot([0,0],[-.001,.05],'--',linewidth=1,color='black')

# # ax.scatter(signal_pred_time_01_pp[:,0],rmse_all_01_pp,marker='1',label=' 0.01 s learning length')
# # ax.scatter(np.array(signal_pred_time_p1)[:,0],rmse_all_p1,marker='o',label='0.1 s learning length')
# # ax.scatter(signal_pred_time_p25_pp[:,0],rmse_all_p25_pp,marker='D',label='0.25 s learning length')
# # ax.scatter((signal_pred_time_p5_pp)[:,0],rmse_all_p5_pp,marker='^',label='  0.5 s learning length')
# # ax.scatter((signal_pred_time_1_pp)[:,0],rmse_all_1_pp,marker='+',label=' 1 s learning length')
# # ax.plot([0,0],[-.01,.05],'--',linewidth=1,color='black')
# ax.yaxis.grid()
# ax.set_axisbelow(True)
# # plt.xlim(0,1)
# # ax.set_yscale('log')
# # plt.semilogy(basey=2)
# plt.xlabel('time (s)')
# plt.ylabel('MAE (m/$s^2$)')
# plt.title('(b)', y=-.7, ha='center')
# # plt.xlim(0,1)
# # plt.grid()
# plt.legend(loc=2,bbox_to_anchor=(0, -0.55),facecolor='white',ncol=4, edgecolor = 'black', framealpha=1)
# # plt.savefig('plots/pointby point_error_one plot_100ms_CT.png', dpi=800)
# # plt.savefig('plots/pointby point_error_one plot_100ms_CT.pdf', dpi=800)
# #%% MAE before and after nonstationary
# fig4 = plt.figure(constrained_layout=True)
# spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig4)
# ax = fig4.add_subplot(spec2[0, 0])
# # ax.scatter(signal_pred_time_01_pp[:,0],mae_all_01_pp,marker='1',label=' 0.01 s learning length')
# ax.scatter(signal_pred_time_p1_pp[:,0],mae_all_p1_pp,marker='o',label='0.1 s learning length')
# # ax.scatter(signal_pred_time_p25_pp[:,0],mae_all_p25_pp,marker='D',label=' 0.25 s learning length')
# ax.scatter((signal_pred_time_p5_pp)[:,0],mae_all_p5_pp,marker='^',label=' 0.5 s learning length')
# ax.scatter((signal_pred_time_1_pp)[:,0],mae_all_1_pp,marker='+',label=' 1 s learning length')
# ax.plot([0,0],[-.001,.035],'--',linewidth=1,color='black')
# ax.yaxis.grid()
# ax.set_axisbelow(True)
# ax.set_yscale('log')
# # plt.semilogy(basey=2)
# plt.xlim(-.75,1.51)
# plt.ylim(0.007,.035)
# plt.xlabel('time (s)')
# plt.ylabel('MAE (m/$s^2$)')
# plt.title('(b)', y=-.35, ha='center')
# plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.30),ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)
# # plt.legend(loc=4,ncol=1,facecolor='white', edgecolor = 'black', framealpha=1)
# # plt.xlim(0,1)
# # plt.grid() 
# # plt.legend(loc=4,facecolor='white', edgecolor = 'black', framealpha=1)#Using matplotlib.pyplot, plt.legend(facecolor='white', framealpha=1) will give your legend a white background without transparency.
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),facecolor='white',ncol=5, edgecolor = 'black', framealpha=1)
# # ax = fig4.add_subplot(spec2[1,0])
# # ax.scatter(signal_pred_time_01_pp[:,0],mae_all_01_pp,marker='1',label=' 0.01 s learning length')
# # ax.scatter(np.array(signal_pred_time_p1)[:,0],mae_all_p1,marker='o',label='0.1 s learning length')
# # ax.scatter((signal_pred_time_p25_pp)[:,0],mae_all_p25_pp,marker='D',label=' 0.25 s learning length')
# # ax.scatter((signal_pred_time_p5_pp)[:,0],mae_all_p5_pp,marker='^',label=' 0.5 s learning length')
# # ax.scatter((signal_pred_time_1_pp)[:,0],mae_all_1_pp,marker='+',label=' 1 s learning length')
# # ax.plot([0,0],[-.001,3.5],'--',linewidth=1,color='black')
# # ax.yaxis.grid()
# # ax.set_axisbelow(True)
# # # ax.set_yscale('log')
# # plt.semilogy(basey=2)
# # plt.xlim(0,5)
# # plt.xlabel('time (s)')
# # plt.ylabel('MAE (m/$s^2$)')
# # plt.title('(b)', y=-.7, ha='center')
# # plt.savefig('plots/pointby point_error_transient_one_coloum plot_100ms_CT.png', dpi=800)
# # plt.savefig('plots/pointby point_error_transient_one_scolumn plot_100ms_CT.pdf', dpi=800)
# # ax.legend(loc='upper center', bbox_to_anchor=(1, -0.2),facecolor='white',ncol=5, edgecolor = 'black', framealpha=1)
# # plt.xlim(0,1)
# # plt.grid() 

# #%% mae with hatch mark

# fig1 = plt.figure(figsize=(6.5,3))
# ax = fig1.add_subplot(211)
# width = .8

# ax.bar(signal_pred_time_p1_pp[:,0]+width/3,mae_all_p1_pp,width/3,hatch='--',label='MAE for 0.1 s learning length')
# ax.bar(signal_pred_time_p5_pp[:,0],mae_all_p5_pp,width/3,hatch='///',label='MAE for 0.5 s learning length')
# ax.bar(signal_pred_time_1_pp[:,0]-width/3,mae_all_1_pp,width/3,hatch='...',label='MAE for 1 s learning length')
# ax.yaxis.grid()
# ax.set_axisbelow(True)
# # plt.xlim(-5,5)
# plt.xlim()
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
# plt.title('(a)', y=-.7, ha='center')
# # plt.xlim(0,1)
# # plt.grid() 
# plt.legend(loc=2,facecolor='white', edgecolor = 'black', framealpha=1)


# # ax = fig1.add_subplot(312)
# # ax.bar(np.array(signal_pred_time_001)[:,0]+width/3,mse_all_001,width/3, label='mse for 0.1 s learning length')
# # ax.bar(signal_pred_time_01_pp[:,0],mse_all_01_pp,width/3,label='mse for 0.5 s learning length')
# # ax.bar(signal_pred_time_1_pp[:,0]-width/3,mse_all_1_pp,width/3,label='mse for 1 s learning length')
# # plt.xlim(-5,5)

# # plt.xlabel('time (s)')
# # plt.ylabel('error (m/$s^2$)')
# # plt.title('(b)', y=-.4, ha='center')
# # plt.xlim(0,1)
# # plt.grid()
# # plt.legend(loc=2)
# ax = fig1.add_subplot(212)

# ax.bar(signal_pred_time_p1_pp[:,0]+width/3,rmse_all_p1_pp,width/3,hatch='--', label='RMSE for 0.1 s learning length')
# ax.bar(signal_pred_time_p5_pp[:,0],rmse_all_p5_pp,width/3,hatch='///',label='RMSE for 0.5 s learning length')
# ax.bar(signal_pred_time_1_pp[:,0]-width/3,rmse_all_1_pp,width/3,hatch='...',label='RMSE for 1 s learning length')
# ax.yaxis.grid()
# ax.set_axisbelow(True)
# # plt.xlim(-5,5)
# plt.xlim()
# plt.xlabel('time (s)')
# plt.ylabel('error (m/$s^2$)')
# plt.title('(b)', y=-.7, ha='center')
# # plt.xlim(0,1)
# # plt.grid()
# plt.legend(loc=2,facecolor='white', edgecolor = 'black', framealpha=1)
# # plt.savefig('plots/mae_mse_rmse_zoom_all_100ms_CT_w_hatch_whole', dpi=1200)
# # plt.savefig('plots/mae_mse_rmse_zoom_all_100ms_CT_w_hatch_whole.pdf', dpi=1200)
# # fig1.tight_layout()
# #%% mae