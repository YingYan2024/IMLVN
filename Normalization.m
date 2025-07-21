function M = Normalization(m, type)
    switch type % 不归一化
        case 1
            M=m;
        case 2  %缩放归一化
            M=(m-min(m,[],2))./(max(m,[],2)-min(m,[],2));
        case 3  %标准归一化
            M=normalize(m,2,'zscore');
        case 4  %中心归一化
            M=normalize(m,2,'scale');
        case 5  %缩放
            Mean=mean(mean(abs(m)));
            M=m/Mean;
        case 6 % PCA
            [M,~,~]=PCA(m);
    end
end