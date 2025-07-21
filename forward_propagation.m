function [output] = forward_propagation(input_data,res_data,ub,W_1,W_2,W_3,Acti_type_1,Acti_type_2,gamma,beta,diff_type,Gaus_type)

        x1 = W_1*input_data;
        xt_1_acti = Activate(x1,Acti_type_1); %Activate为激活函数
        et = Edifference(res_data,xt_1_acti,diff_type);

        % 变换为正态分布
        [et_Gau, ~, ~, ~] = Gaussian_trans(et,gamma,beta,Gaus_type);
    
        st = 3-(4./(1+exp(1).^(-et_Gau*log(3)./(ub))));
        % st = et_Gau;
    
        X2 = W_2 * st;
        X2_acti = Activate(X2,Acti_type_2);
        
        X3 = W_3 * X2_acti;
    
        %Vst_acti = (1+exp(1).^(Vst)).^(-1);
        X3_acti = -1 * X3;
        output = softmax(X3_acti);

end