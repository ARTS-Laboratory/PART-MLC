clc;
clear all;
%% Load data
% data=xlsread('FFT_matlab\nonstationarity_data.csv');
% time=data(:,1); % 1st column
% X_data=data(:,2);% 2nd column
data = load('FFT_matlab\data.mat');
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
% Y_pred = fft_prediction(xx, dt, forcast_horizon_steps,freq_list,true);
%% partMLC
X=xx;
returnVector=true;
% Set up new time related items. 
tt = 0:dt:(size(X,2)*dt)-dt;
Ts=(tt(end)-tt(1))/size(tt,2);

% Detrending and taking FFT
n = size(X,2);
t1 = 1:n;
p = polyfit(t1, X, 1); % find the trend
trend=p(2);
x_notrend = X- (trend* t1(1,:));  % detrended x in frequency domain

%% building a new input length which will be increase or decrease by power of two
q = 0; % input length  controlling parameter
if n<1/Ts
    while (n - (2 ^ q) * (1 / Ts)) < 0
        q=q-1;
        if q<-1
            fprintf("The input length must be greater than the half of the sample rate for better prediction");
        end
    end
elseif n> 1/Ts
    while (n - (2 ^ q) * (1 / Ts)) > (2 ^ q) * (1 / Ts)
        q=q+1;
    end
else      
    % catches the n=1/Ts, in that case q=0
end
n_prime = floor((2 ^ q) * (1 / Ts)); % new input length % Alert: fix -1 later.
x_freqdom = fft(x_notrend(1:n_prime)); % detrended x in frequency domain
f = fftfreq(size(x_freqdom,2), Ts);  % frequencies

%% build the index of freqency values
freq_idx=[];
if isempty(freq_list)
    fprintf("No frequency list has been provided. Running for all the frequencies.")
    freq_list = f; %If no frequencies provided, it will return a warning and use all frequences.
    freq_idx = 1:size(freq_list,2);
else
    for i=1:size(freq_list,2)
        b = find(abs(f-freq_list(i))<0.1); 
        freq_idx(end+1)=b;
    end
end

%% creates vecotrs for rebuilding the signal
t2 = 1:(n + forcast_horizon_steps);
restored_sig = zeros(1,size(t2,2));
% rebuild the signal
for j=1:size(freq_idx,2)
    ampli = abs(x_freqdom(freq_idx(j))) /n_prime;  % find amplitude
    phase = angle(x_freqdom(freq_idx(j)));  % find phase
    freq_rad_per_sec = 2 * pi * (f(freq_idx(j))); % report the frequency in rad for the cos function
    freq_rad_per_sample = freq_rad_per_sec*Ts; % returns frequency in rad_per_sample.Reason: f[j]*Ts: to make sampling frequency cause only f[j] is just frequency
    restored_sig = restored_sig + ampli * cos(freq_rad_per_sample * t2 + phase); % restored signal with respect to phase,amplitude and frequencies
end
% add trend

Y = restored_sig +(trend * t2);

if returnVector
    Y_pred = Y(size(X,2):end);
else
    Y_pred = Y(end);
end









