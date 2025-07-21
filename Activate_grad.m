function [X] = Activate_grad(x,type)
    switch type
        case 1 %RELU
            X=ones(size(x));
            X(x<=0)=0;
        case 2 %LeakyRELU
            X=ones(size(x));
            X(x<=0)=0.01;
        case 3 %Logistic
            tem = 1./(1+exp(-x));
            X = tem.*(1-tem);
        case 4 %tanh
            X = sech(x).^2;
        case 5 % None
            X = ones(size(x));
        case 6 %softsign
            X = 1 ./ (1 + abs(x)).^2;
        case 7 %softplus
            X = 1 ./ (1 + exp(-x));
        case 8 %ELU
            alpha = 1; % 可以根据需要调整alpha的值
            X = (x > 0) + (x <= 0) .* (alpha + x);
        case 9 %Swish
            beta = 1; % 可以根据需要调整beta的值
            sigmoid_beta_x = 1 ./ (1 + exp(-beta * x));
            X = sigmoid_beta_x + x .* sigmoid_beta_x .* (1 - sigmoid_beta_x);
    end
end
