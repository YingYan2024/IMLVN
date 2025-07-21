function [output,d_gamma,d_beta] = Gaussian_trans_grad(d_output, input, normalized_input, gamma, mean_val, var_val,type)
    switch type
        case 1
            output = d_output .* ones(size(input));
            d_gamma=0;
            d_beta=0;
        case 2
            output = d_output .* (input + 1).^(-1);
            d_gamma=0;
            d_beta=0;
        case 3
            [num_features, ~] = size(input);
        
            d_gamma = sum(d_output .* normalized_input, 2);
            d_beta = sum(d_output, 2);
        
            d_normalized = d_output .* gamma;
        
            d_var = sum(d_normalized .* (input - mean_val) .* (-0.5) .* (var_val + 1e-8).^(-1.5), 1);
            d_mean = sum(d_normalized .* (-1) ./ sqrt(var_val + 1e-8), 1) + d_var .* sum(-2 .* (input - mean_val), 1) / num_features;
        
            output = d_normalized ./ sqrt(var_val + 1e-8) + d_var .* 2 .* (input - mean_val) / num_features + d_mean / num_features;
        case 4
            [~, W] = size(input);

            d_gamma = sum(d_output .* normalized_input, 2);
            d_beta = sum(d_output, 2);
        
            d_normalized = d_output .* gamma;
        
            d_var = sum(d_normalized .* (input - mean_val) .* (-0.5) .* (var_val + 1e-8).^(-1.5), 2);
            d_mean = sum(d_normalized .* (-1) ./ sqrt(var_val + 1e-8), 2) + d_var .* sum(-2 .* (input - mean_val), 2) / W;
        
            output = d_normalized ./ sqrt(var_val + 1e-8) + d_var .* 2 .* (input - mean_val) / W + d_mean / W;
        case 5
            output = 0.5 * d_output .* input.^(-0.5);
            d_gamma=0;
            d_beta=0;
    end
end