function [output, normalized_input, mean_val, var_val] = Gaussian_trans(input,gamma,beta,type)
    switch type
        case 1
            output = input;
            mean_val=0;
            var_val=0;
            normalized_input=0;
        case 2
            output = log(input + 1);
            mean_val=0;
            var_val=0;
            normalized_input=0;
        case 3
            mean_val = mean(input, 1); % 计算整个批次的均值
            var_val = var(input, 0, 1); % 计算整个批次的方差
        
            normalized_input = (input - mean_val) ./ sqrt(var_val + 1e-8); % 归一化
            output = gamma .* normalized_input + beta; % 缩放和平移
        case 4
            mean_val = mean(input, 2); % 计算每个样本的均值
            var_val = var(input, 0, 2); % 计算每个样本的方差
        
            normalized_input = (input - mean_val) ./ sqrt(var_val + 1e-8); % 归一化
            output = gamma .* normalized_input + beta; % 缩放和平移
        case 5
            output = sqrt(input);
            mean_val=0;
            var_val=0;
            normalized_input=0;
    end
end