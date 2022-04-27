clc;
clear all;
%% Load data
% data=xlsread('FFT_matlab\nonstationarity_data.csv');
% time=data(:,1); % 1st column
% X_data=data(:,2);% 2nd column
data = load('data.mat');
time=data.data(:,1)'; % 1st column
X_data=data.data(:,2)';% 2nd column
plot(X_data,time);
title('data');
xlabel('time (data points)');
ylabel('acceleration (g)');
dt=1.9531228885135136e-05;

% User defined parameters
freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,-20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240];
% sorted  frequencies with more impact on the original FFT
forcast_horizon_steps= 5120; % prediction length # here 1s=51200 samples/sec

% Input length should capture the minimum frequency. 
xx_length = 29000;
xx = X_data(1:xx_length);
%% Running the function
Y_pred = fft_prediction(xx, dt, forcast_horizon_steps,freq_list,true);
figure(2)
plot(Y_pred);










