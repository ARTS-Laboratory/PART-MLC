function[dws_err_min,optimal_ws_t]=get_fft_dws_t(dws_t_init,dws_t_max,t_pred,data_t_sbsd,data_acc_sbsd)
    %inputs: initial window size (minimum) (s), maximum allowable window
    %size (s), time data (x axis) (array, s), acceleration data (y axis) (array,s);
    %input arrays are subarrays of bigger arrays, should hold that
    % lenght is floor(dws_t_max/sbsl_period)+round(t_pred/sbsl_period)+1;
    sbsl_period=data_t_sbsd(2)-data_t_sbsd(1);
    err_accum_average=zeros(1,floor((dws_t_max-dws_t_init)/(sbsl_period))+1);
    if (dws_t_init/(sbsl_period))<5
        disp('err: N initial for FFT window size is less than min (5)');
    
    end
    if (t_pred/(sbsl_period))<5
        disp('err: N prediction samples is less than min (5)');
    
    end

    optimal_ws_t=dws_t_init; dws_err_min=max(data_acc_sbsd)-min(data_acc_sbsd);
    for curr_dws_tt=dws_t_init:(sbsl_period):dws_t_max
        [X_curr,freq_curr]=positiveFFT(data_acc_sbsd(1:floor(curr_dws_tt/sbsl_period)),1/(sbsl_period));
        acc_ifft_sbsd=ifft(X_curr);
%         if floor(curr_dws_tt/sbsl_period)==157
%             disp('here');
% 
%         end
%         if floor(curr_dws_tt/sbsl_period)>=157
%         figure(1); plot(data_t_sbsd,data_acc_sbsd,'.',data_t_sbsd(1:floor(curr_dws_tt/sbsl_period)),data_acc_sbsd(1:floor(curr_dws_tt/sbsl_period)),'m'); grid on; hold on;
%         end
        for curr_pred_t=0:(sbsl_period):t_pred %this is inside ifft array;
            curr_tt_nmzd=mod(curr_pred_t,data_t_sbsd(floor(curr_dws_tt/sbsl_period)+1)-data_t_sbsd(1));
            [~,clst_idx_val]=min(abs(data_t_sbsd(1:length(acc_ifft_sbsd))-data_t_sbsd(1)-curr_tt_nmzd));
            err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)= ...
                err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)+ ...
                abs(data_acc_sbsd(floor((curr_pred_t+curr_dws_tt)/sbsl_period)+1)- ...
                acc_ifft_sbsd(clst_idx_val));
%             if floor(curr_dws_tt/sbsl_period)>=157
%             plot(data_t_sbsd(floor(curr_dws_tt/sbsl_period)+round(curr_pred_t/sbsl_period)+1),acc_ifft_sbsd(clst_idx_val)','r*');
%             plot(data_t_sbsd(floor(curr_dws_tt/sbsl_period)+round(curr_pred_t/sbsl_period)+1),data_acc_sbsd(floor((curr_pred_t+curr_dws_tt)/(sbsl_period))+1),'g*');
%             end
    
        end
        if err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)/(floor(t_pred/sbsl_period)+1)<dws_err_min
            dws_err_min=err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)/(floor(t_pred/sbsl_period)+1);
            optimal_ws_t=floor(curr_dws_tt/sbsl_period)*sbsl_period;
    
        end
%         if floor(curr_dws_tt/sbsl_period)>=157
%         hold off;
%         end
        err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)= ...
                err_accum_average(round((curr_dws_tt-dws_t_init)/(sbsl_period))+1)/(floor(t_pred/sbsl_period)+1);
    
    end
    figure(33); stem(floor(dws_t_init/sbsl_period)+1:1:floor(dws_t_max/sbsl_period),err_accum_average); grid on;


end

function[X,freq] = positiveFFT(x,Fs)
    freq=(0:length(x)-1)*Fs/length(x);
    X = fft(x);
end