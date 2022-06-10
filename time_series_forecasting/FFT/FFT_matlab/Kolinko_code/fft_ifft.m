clear; %clear workspace;
%clc; %clear cammand window;
%close all; %close all figures whose handles are visible;

data_fl_nm="data/data_set_3.lvm";

data_lvm=lvm_import(data_fl_nm); %original data;
data_t=data_lvm.Segment1.data(:,1);
data_acc=data_lvm.Segment1.data(:,4);
figure(1)
plot(data_t,data_acc) %plot acceleration(time);
sample_rate=1/data_lvm.Segment1.Delta_X(1);

subsample_in_val=20;
subsample_out_val=1;
data_t_sbsd=data_t(1):((data_t(size(data_t,1))-data_t(1))/((size(data_t,1)/subsample_in_val)*subsample_out_val)):data_t(size(data_t,1));
data_t_sbsd=data_t_sbsd';%to be consistent with columns/rows order in vector;
data_acc_sbsd=zeros(size(data_t_sbsd,1),1);
for ii=1:size(data_t_sbsd,1)
    [cls_val_t,cls_idx]=min(abs(data_t_sbsd(ii)-data_t));
    if (data_t_sbsd(ii)-data_t(cls_idx))<0
        curr_weight=(data_t(cls_idx)-data_t_sbsd(ii))/(data_t(cls_idx)-data_t(cls_idx-1));
        data_acc_sbsd(ii)=data_acc(cls_idx-1)*curr_weight+data_acc(cls_idx)*(1-curr_weight);

    elseif(data_t_sbsd(ii)-data_t(cls_idx))>0
        curr_weight=(data_t_sbsd(ii)-data_t(cls_idx))/(data_t(cls_idx+1)-data_t(cls_idx));
        data_acc_sbsd(ii)=data_acc(cls_idx)*(1-curr_weight)+data_acc(cls_idx+1)*curr_weight;
    else
        data_acc_sbsd(ii)=data_acc(cls_idx);
    end

end
figure(2)
plot(data_t,data_acc,'b--',data_t_sbsd,data_acc_sbsd,'m') %plot acceleration(time) for subsampled signal;



%%
freqs_decl=[100,120];
freqs_decl_lcm=1;%least common multiple of declared frequencies (a_1*a_2*...*a_n*x1);
for ii=1:length(freqs_decl)
    freqs_decl_lcm=lcm(freqs_decl_lcm,freqs_decl(ii));
end
freqs_decl_cpd=1;%product of koefficiints (a_1*a_2*...*a_n*b_1*b_2*...*z_n);
for ii=1:length(freqs_decl)
    freqs_decl_cpd=freqs_decl_cpd*freqs_decl_lcm/freqs_decl(ii);
end
t_decl_min_fft=freqs_decl_cpd/freqs_decl_lcm;

[cls_val_t_sd,cls_idx_sd]=min(abs(t_decl_min_fft-data_t_sbsd));
if((data_t_sbsd(cls_idx_sd)-t_decl_min_fft)<0)
    cls_idx_sd=cls_idx_sd+1;
end

[X_fft_sbsd,freqs_fft]=positiveFFT(data_acc_sbsd(1:cls_idx_sd),1/(data_t_sbsd(cls_idx_sd)-data_t_sbsd(cls_idx_sd-1)));
figure(3);
stem(freqs_fft,abs(X_fft_sbsd));
acc_ifft_sbsd=ifft(X_fft_sbsd);
figure(4);
plot(data_t_sbsd(1:cls_idx_sd),data_acc_sbsd(1:cls_idx_sd),'c.',data_t_sbsd(1:cls_idx_sd),acc_ifft_sbsd);

hold on
t_pred=5;%prediction time;
t_pred=t_pred-floor(t_pred/data_t_sbsd(cls_idx_sd))*data_t_sbsd(cls_idx_sd)%"substract" from t_pred time which corresponds to full periods;
[cls_val_t_pred,cls_idx_pred]=min(abs(t_pred-data_t_sbsd(1:cls_idx_sd)));
plot(data_t_sbsd(cls_idx_pred),acc_ifft_sbsd(cls_idx_pred),'r*');
hold off
%%
init_fft_samples=10; increment_samples=1;
max_fft_samples=1000;%maximum allowable FFT window size; if no optimal window size is found, reset;
err_accum_average=zeros(1,((max_fft_samples-init_fft_samples)/increment_samples)+1);
pred_samples=10;
figure(35);plot(data_t_sbsd(1:max_fft_samples),data_acc_sbsd(1:max_fft_samples));
for curr_samples=init_fft_samples:increment_samples:max_fft_samples
    [X_curr,freq_curr] = positiveFFT(data_acc_sbsd(1:curr_samples),1/(sbsl_period));
    %figure(33); stem(freq_curr,abs(X_curr));

    acc_sbsd_ifft=ifft(X_curr);
    %figure(34);plot(data_t_sbsd(1:max_fft_samples),data_acc_sbsd(1:max_fft_samples),':',data_t_sbsd(1:length(acc_sbsd_ifft)),acc_sbsd_ifft,'--');

    for ii=(curr_samples+1):1:(curr_samples+1+pred_samples)
        curr_t_pred=mod(data_t_sbsd(ii),data_t_sbsd(length(acc_sbsd_ifft)));
        [clst_t_val,clst_idx_val]=min(abs(curr_t_pred-data_t_sbsd(1:length(acc_sbsd_ifft))));
        err_accum_average(((curr_samples-init_fft_samples)/increment_samples)+1)= ...
            err_accum_average(((curr_samples-init_fft_samples)/increment_samples)+1)+abs(acc_sbsd_ifft(clst_idx_val)-data_acc_sbsd(ii));
    end
    err_accum_average(((curr_samples-init_fft_samples)/increment_samples)+1)=...
        err_accum_average(((curr_samples-init_fft_samples)/increment_samples)+1)/pred_samples;
end

figure(5); stem(init_fft_samples:increment_samples:max_fft_samples,err_accum_average);
%%
strt_sample=1; sbsl_period=data_t_sbsd(2)-data_t_sbsd(1);
dws_t_init=2*10^(-3); dws_t_max=0.3;
err_accum_average=zeros(1,floor((dws_t_max-dws_t_init)/(sbsl_period))+1);
if (dws_t_init/(sbsl_period))<5
    disp('err: N initial for FFT window size is less than min (5)');

end
t_pred=0.1; t_pred; t_pred; cnt_tmp=1;
if (t_pred/(sbsl_period))<5
    disp('err: N prediction samples is less than min (5)');

end

for curr_dws_tt=dws_t_init:(sbsl_period):dws_t_max
    [X_curr,freq_curr]=positiveFFT(data_acc_sbsd(strt_sample:strt_sample+floor(curr_dws_tt/(sbsl_period))+1), ...
        1/(sbsl_period));
%     figure(33); stem(freq_curr,abs(X_curr)); grid on;

    acc_ifft_sbsd=ifft(X_curr);
%     figure(34); hold on;
%     plot(data_t_sbsd(1:strt_sample+floor(dws_t_max/(sbsl_period))+1), ...
%         data_acc_sbsd(1:strt_sample+floor(dws_t_max/(sbsl_period))+1),'b.');
%     plot(data_t_sbsd(1:strt_sample+floor(curr_dws_tt/(sbsl_period))+1), ...
%         [zeros(strt_sample-1,1); acc_ifft_sbsd],'m');
    
    for curr_pred_t=0:(sbsl_period):t_pred %this is inside ifft array;
        curr_pred_tt=curr_dws_tt+curr_pred_t;
        curr_tt_nmzd=mod(curr_pred_tt,curr_dws_tt)+(strt_sample-1)*sbsl_period;
        [clst_t_val,clst_idx_val]=min(abs(data_t_sbsd(strt_sample:strt_sample+length(acc_ifft_sbsd))-curr_tt_nmzd));
        err_accum_average(floor((curr_dws_tt-dws_t_init)/(sbsl_period)+0.5)+1)= ...
            err_accum_average(floor((curr_dws_tt-dws_t_init)/(sbsl_period)+0.5)+1)+ ...
            abs(data_acc_sbsd(strt_sample+floor(curr_pred_tt/(sbsl_period))+2)- ...
            acc_ifft_sbsd(clst_idx_val));
%         plot(data_t_sbsd(strt_sample+floor(curr_pred_tt/(sbsl_period))+2),acc_ifft_sbsd(clst_idx_val)','r*');
%         plot(data_t_sbsd(strt_sample+floor(curr_pred_tt/(sbsl_period))+2),data_acc_sbsd(strt_sample+floor(curr_pred_tt/(sbsl_period))+2),'g*');

    end
%     hold off;
%     err_accum_average(floor((curr_dws_tt-dws_t_init)/(sbsl_period))+1)= ...
%             err_accum_average(floor((curr_dws_tt-dws_t_init)/(sbsl_period))+1)/(t_pred/(sbsl_period));

end
figure(3); stem(floor(dws_t_init/sbsl_period)+2:1:floor(dws_t_max/sbsl_period)+1,err_accum_average); grid on;
%%
strt_sample=100; dws_t_init=2*10^(-3); dws_t_max=0.3; t_pred=0.1; sbsl_period=data_t_sbsd(2)-data_t_sbsd(1);




















[dws_err_min,optimal_ws_t]=get_fft_dws_t(dws_t_init,dws_t_max,t_pred, ... 
    data_t_sbsd(strt_sample:strt_sample+floor(dws_t_max/sbsl_period)+round(t_pred/sbsl_period)), ...
    data_acc_sbsd(strt_sample:strt_sample+floor(dws_t_max/sbsl_period)+round(t_pred/sbsl_period)));
%at this point t is approximately curr_t_sbsd=strt_sample*sbsl_period+t_pred+dws_t_max;
%so prediction should start from sample curr_t_sbsd;
[X_fft_opt,freq_opt]=positiveFFT(data_acc_sbsd(strt_sample:strt_sample+floor(optimal_ws_t/sbsl_period)-1),1/(sbsl_period));
figure(3); stem(freq_opt,abs(X_fft_opt)); grid on;
iff_opt_sbsd=ifft(X_fft_opt);
figure(4); plot(data_t_sbsd(strt_sample:length(iff_opt_sbsd)+strt_sample-1),iff_opt_sbsd); grid on;

curr_t_sbsd=data_t_sbsd(strt_sample+floor(dws_t_max/sbsl_period)+round(t_pred/sbsl_period));
% curr_t_sbsd=data_t_sbsd((optimal_ws_t*5)/sbsl_period);
figure(5); hold on; plot(data_t_sbsd(strt_sample:strt_sample+floor(optimal_ws_t/sbsl_period)-1),iff_opt_sbsd,'.', ...
        data_t_sbsd(1:round(curr_t_sbsd/sbsl_period)),data_acc_sbsd(1:round(curr_t_sbsd/sbsl_period))); grid on;
v_tmp=zeros(size(iff_opt_sbsd));

for curr_t_pred=curr_t_sbsd-(floor(t_pred/sbsl_period)-1)*sbsl_period:sbsl_period:curr_t_sbsd
    curr_t_pred_nmzd=mod(curr_t_pred-(strt_sample-1)*sbsl_period,optimal_ws_t);
    [cls_t_val,idx_cls_t]=min(abs(data_t_sbsd(strt_sample:strt_sample+floor(optimal_ws_t/sbsl_period)-1)-data_t_sbsd(strt_sample)-curr_t_pred_nmzd));
    plot(data_t_sbsd(round(curr_t_pred/sbsl_period)),iff_opt_sbsd(idx_cls_t),'r*');

    v_tmp(round((curr_t_pred-(curr_t_sbsd-(floor(t_pred/sbsl_period)-1)*sbsl_period))/sbsl_period)+1)=idx_cls_t;
    
    %err_calc;







end
hold off;
figure(10); stairs(v_tmp);






%%
function[X,freq] = positiveFFT(x,Fs)
    freq=(0:length(x)-1)*Fs/length(x);
    X = fft(x);
end