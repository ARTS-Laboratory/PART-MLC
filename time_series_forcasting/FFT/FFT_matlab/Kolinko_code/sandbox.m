%%
close all;

f1=11; f2=5; %frequencies of two signals;
tt=0:0.01:10;
f_sum=sin(2*pi*tt)+sin(1/(2*pi)*tt);%f_sum=sawtooth(1/(2*pi)*tt)+square(1/(2*pi)*sqrt(11)*tt);%
plot(tt,f_sum)
%commulative frequency is 11/11=1; (a*x1=b*x2; f_c=x1/b=x2/a);
% g = fittype('a+b*x+c*x*x');
% model=fit(tt',f_sum',g); plot(tt,f_sum,':',tt,model.a+model.b*tt+model.c*tt.*tt,'--',tt,f_sum-(model.a+model.b*tt+model.c*tt.*tt))
%%
close all;

fo = 4; %frequency of the sinewave
Fs = 100; %sampling rate 
Ts = 1/Fs; %sampling time interval
t = 0:Ts:2; 
%n = length(t); %number of samples 
y = sin(2*pi*fo*t)+cos(2*pi*5*fo*t);
figure(1)
plot(t,y,'b')

grid on
[YfreqD,freqRng] = positiveFFT(y(1:floor(length(y)/8)),Fs);
figure(2)
% stem(freqRng,abs(YfreqD));
stem(freqRng,abs(YfreqD));

B = ifft(YfreqD);%*(length(y)/8);
t1 = (0:Ts:Ts*(length(B)-1));
figure(3)
plot(t,y,'b:',t1,B,'--')
grid on
%%
close all;

Fs = 100; %sampling rate
tt = 0:1/Fs:5; 
%n = length(t); %number of samples 
y = sin(2*pi*tt)+cos(2*pi*5*tt);
figure(1)
plot(tt,y,'b')

init_fft_samples=10; increment_samples=2;
err_accum_average=zeros(size(init_fft_samples:increment_samples:floor(length(y)/2)));
prediction_time=tt(length(tt))/2;

for curr_fft_samples=init_fft_samples:increment_samples:floor(length(y)/2)
    [X_curr,freq_curr]=positiveFFT(y(1:curr_fft_samples),Fs);
    %figure(1); stem(freq_curr,abs(X_curr)); grid on;

    B_curr=ifft(X_curr);
    %figure(2); plot(tt,y,':',tt(1:length(B_curr)),B_curr,'--'); grid on

    arr_accum_curr=0; %zeros(size(B_curr));
    for local_tt=tt(curr_fft_samples):1/Fs:tt(curr_fft_samples)+prediction_time
        reff_tt=mod(local_tt,tt(length(B_curr)));
        [abs_min_reff,idx_reff]=min(abs(reff_tt-tt));
        [abs_min_tt,idx_tt]=min(abs(local_tt-tt));
        %arr_accum_curr(idx_reff)=abs(y(idx_tt)-B_curr(idx_reff));
        %figure(3); plot(tt(1:length(B_curr)),arr_accum_curr); grid on;
        arr_accum_curr=arr_accum_curr+abs(y(idx_tt)-B_curr(idx_reff));





    end
    arr_accum_curr=arr_accum_curr/((prediction_time-tt(1))/(1/Fs));
    (curr_fft_samples-init_fft_samples)/increment_samples;
    err_accum_average(((curr_fft_samples-init_fft_samples)/increment_samples)+1)=arr_accum_curr;


end
figure(4); stem(init_fft_samples:increment_samples:floor(length(y)/2),err_accum_average)



%%

function[X,freq] = positiveFFT(x,Fs)
N = length(x);
k = 0:N-1;
T = N/Fs;
freq = k/T; %create the frequency range
X = fft(x);%/N; %normalize the data 
%cutOff = N;%ceil(N/2);
%X = X(1:cutOff);
%freq = freq(1:cutOff);
end
