function [output] = Edifference(res,x,type)
    switch type
        case 1 % 绝对值误差
            output=abs(res-x);
        case 2 % 均方误差
            output=(res-x).^2;
        case 3 % huber损失
            delta=1; % 设置阈值
            diff=abs(res-x);
            % 对差值应用Huber损失
            squared_loss = 0.5 * diff.^2;
            absolute_loss = delta * abs(diff) - 0.5 * delta^2;
            
            % 使用逻辑索引确定哪些元素应该使用哪种损失
            use_squared = abs(diff) <= delta;
            
            % 初始化输出矩阵并填充相应的损失值
            output = zeros(size(diff));
            output(use_squared) = squared_loss(use_squared);
            output(~use_squared) = absolute_loss(~use_squared);
        case 4 % 直接相减
            output = res - x;
        case 5 % KL散度
            output = (res * log(res ./ x) + (1 - res) * log((1 - res) ./ (1 - x)));
    end
end