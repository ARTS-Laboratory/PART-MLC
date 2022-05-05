% input_x=1:20; output_y=1:2:40; n_tgt=10;
% [output_x,output_y]=bilin_resample1(input_x,input_y,n_tgt);

function [output_x,output_y]=bilin_resample(input_x,input_y,n_tgt)
    %bilin_resample can use weighted average to change input_x and and
    %input_y to contain n_tgt samples; source data is in input_x,input_y,
    %which should hold (x,y) correspondance
    %(length(input_x)==length(input_y));
    %if length(input_x)<n_tgt, superserpling performed
    %(length(output_x)==length(output_y)==n_tgt >
    %length(input_x)==length(input_y)); otherwise subsampling can be done;
    






    output_x=input_x(1):(input_x(length(input_x))-input_x(1))/(n_tgt-1):input_x(length(input_x));
    output_y=zeros(size(output_x));
    for ii=1:1:length(output_x)
        [~,cls_idx]=min(abs(output_x(ii)-input_x));
        if (output_x(ii)-input_x(cls_idx))<0
            cls_idx1=cls_idx-1;
            curr_weight=(output_x(ii)-input_x(cls_idx1))/(input_x(cls_idx)-input_x(cls_idx1));
            output_y(ii)=input_y(cls_idx1)*(1-curr_weight)+input_y(cls_idx)*curr_weight;

        elseif (output_x(ii)-input_x(cls_idx))>0
            cls_idx1=cls_idx+1;
            curr_weight=(output_x(ii)-input_x(cls_idx))/(input_x(cls_idx1)-input_x(cls_idx));
            output_y(ii)=input_y(cls_idx)*(1-curr_weight)+input_y(cls_idx1)*curr_weight;

        else
            output_y(ii)=input_y(cls_idx);
            
        end

            

    end


end