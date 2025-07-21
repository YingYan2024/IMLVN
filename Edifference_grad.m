function [output] = Edifference_grad(res,x,type,d_et)
    switch type
        case 1
            d_abs = sign(res - x); % 用于绝对值的梯度计算
            output = d_et .* d_abs * -1;
        case 2
            output = d_et .* (res - x) * -2;
        case 4
            output = x;
    end
end