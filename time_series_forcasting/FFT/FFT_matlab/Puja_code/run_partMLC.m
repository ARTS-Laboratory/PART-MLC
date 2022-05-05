clc;
clear all;
%% Load data
% data=xlsread('FFT_matlab\nonstationarity_data.csv');
data = load('data.mat');
X_data=data.data(:,1)';% 1st column acceleration
dt=1.9531228885135136e-05;

% User defined parameters
freq_list = [20,60,70,80,100,120,140,150,160,170,180,200,220,240,-20,-60,-70,-80,-100,-120,-140,-150,-160,-170,-180,-200,-220,-240];
% sorted  frequencies with more impact on the original FFT
forcast_horizon_steps= 5120; % prediction length # here 1s=51200 samples/sec

% Input length should capture the minimum frequency. 
xx_length = 5120; 
xx = X_data(1:xx_length);
%% Running the function
% y1_withoutFreq =fft_prediction(xx,dt,forcast_horizon_steps,[],true); % Returns the vector of data up to forcast_horizon_steps if returnVector=True
y1_withFreq = fft_prediction(xx,dt,forcast_horizon_steps,freq_list,true); % Returns the vector of data up to forcast_horizon_steps if returnVector=True
% y2_withoutFreq = fft_prediction(xx,dt,forcast_horizon_steps,[],false);  % Just  returns 1 point forcast_horizon_steps into the future if returnVector=False
y2_withFreq = fft_prediction(xx,dt,forcast_horizon_steps,freq_list,false); %Just returns 1 point forcast_horizon_steps into the future if returnVector=False

%% plot the code
%%%% plot for with frequency list
figure(2)
forecast = cat(2,NaN(1,xx_length),y1_withFreq);
plot(X_data(1:size(forecast,2)),'DisplayName','truth');
hold on
plot(forecast,'DisplayName','forcast');
plot(xx,'DisplayName','training data')
title('With Freq');
xlabel('time (data points)');
ylabel('acceleration (g)');
hold off
legend
%% Plot for without frequency
% figure(2)
% forecast = cat(2,NaN(1,xx_length),y1_withoutFreq);
% plot(X_data(1:size(forecast,2)),'DisplayName','truth');
% hold on
% plot(forecast,'DisplayName','forcast');
% plot(xx,'DisplayName','training data')
% title('Without Freq');
% xlabel('time (data points)');
% ylabel('acceleration (g)');
% hold off
% legend
% 

