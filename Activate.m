function [X,Type] = Activate(x,type)
    switch type
        case 1 %RELU
            X = max(0,x);
        case 2 %LeakyRELU
            X = max(0.01*x,x);
        case 3 %Logistic
            X = 1./(1+exp(-x));
        case 4 %tanh
            X = tanh(x);
        case 5 % None
            X = x;
        case 6 %softsign
            X = x ./ (1 + abs(x));
        case 7 %softplus
            X = log(1 + exp(x));
        case 8 %ELU
            alpha = 1; % 可以根据需要调整alpha的值
            X = (x > 0) .* x + (x <= 0) .* (alpha * (exp(x) - 1));
        case 9 %Swish
            beta = 1; % 可以根据需要调整beta的值
            X = x ./ (1 + exp(-beta * x));
    end
    Type=type;
end
