function Y_pred = fft_prediction(X,dt,forcast_horizon_steps,freq_list,returnVector)
%FFT_PREDICTION This function used FFT based model to predict the series data.
%{ 
This function used FFT based model to predict the series data.

Parameters
----------
X : numpy.ndarray
    This model is taking X data. But any series data can be used instead of X.
dt : float
    time difference between two sample points.
forcast_horizon : integer
    length of prediction.
returnVector: boolean
     Returns the vector of data up to forcast_horizon_steps if returnVector=True
     Just returns 1 point forcast_horizon_steps into the future if returnVector=False
freq_list: list
    sorted  frequencies with more impact on the original FFT. In no freq_list 
    provided, runns for all frequencies.
Returns
-------
Y : List if returnVector==True, else Float
    If returnVector==True, then it will return the predicted vector,
    otherwise it will return the last sample point of the predicted vector.

%}



% Set up new time related items.
tt = 0:dt:(size(X,2)*dt)-dt;
Ts=(tt(end)-tt(1))/size(tt,2);

% Detrending and taking FFT
n = size(X,2);
t1 = 1:n;
p = polyfit(t1, X, 1); % find the trend
trend=p(1);
x_notrend = X- (trend* t1(1,:));  % detrended x in frequency domain

%% building a new input length which will be increase or decrease by power of two
q = 0; % input length  controlling parameter
if n<1/Ts
    while (n - (2 ^ q) * (1 / Ts)) < 0
        q=q-1;
        if q<-1
            fprintf("The input length must be greater than the half of the sample rate for better prediction\n");
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
