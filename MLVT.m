clc;clear;

rng(1);

t0=tic; % Timer for the entire process

% Set learning rate and training epochs
testNum=1; % Number of repeated experiments
numEpochs = 1000; % Maximum iterations per fold in each experiment's 10-fold cross-validation
lr = 0.005; % Initial learning rate
k=10;  % 10-fold cross-validation
Acti_type_1 = 4; % Activation function type: 1RELU, 2LeakyRELU, 3Logistic, 4tanh, 5None, 6softsign, 7softplus, 8ELU, 9Swish
Acti_type_2 = 2;
diff_type = 2; %1Absolute error, 2Mean squared error, 3Huber loss, direct subtraction
Gaus_type = 1; % 1None, 2Log transform, 3BN, 4LN, 5Root
Power = 2; % Highest expansion order for MTN
class_num = 14; % Number of classes
shift_num = 0.3;

%%%%%%%%%%%%%%%%%%% Data Import %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Data
format long

% feature = feature(:, 1:end-540);

total_num=size(feature,2);
len_data=size(feature,1);
train_num=round(total_num*(k-1)/k); % Training set size (9-fold quantity)
one_fold=total_num/class_num;

Labels=zeros(1,total_num);
for i=1:class_num
    Labels(((i-1)*one_fold+1):(i*one_fold))=i;
end

% Set network structure
feature_size=size(feature,1); % Dimension of feature vector

expan_size=1;
for i=1:Power
    expan_size=expan_size + prod(feature_size:(feature_size+i-1))/prod(1:i); % Dimension after polynomial expansion
end

hidden_size = feature_size;
output_size = class_num;

% One-hot encoding
onehot = zeros(train_num, class_num); % Initialize one-hot encoding matrix
for i = 1:total_num
    onehot(i, Labels(i)) = 1;
end
onehot = onehot'; % Transpose matrix to meet specific format requirements

figure; % Create a new figure window
h1 = animatedline('Color','b'); % Create an animated line for training set accuracy
h2 = animatedline('Color','r'); % Create an animated line for test set accuracy
xlabel('Epoch');
ylabel('Accuracy');
xlim([0 numEpochs]); % Set x-axis range from 0 to max iterations
ylim([0 1]); % Set y-axis range from 0 to 1 (since accuracy ranges from 0 to 1)

Total_acc=0; % Total accuracy count
Total_conf_acc=zeros(class_num, class_num); % Total confusion matrix count

% Et = zeros(1000,10,6318);

for testingNum=1:testNum  % Multiple repeated experiments
    accuracy = zeros(1, k);
    confusionMatrix=zeros(class_num,class_num,k);
    class_acc = zeros(class_num, 1); % Calculate accuracy for each class

    % Cross-validation implementation ensures equal samples per class in each fold and shuffles dataset
    cv = cvpartition(Labels', 'kfold', k); 
    for i=1:k  % k-fold cross-validation
        res_Train_Data=feature(:,cv.training(i));
        res_Test_Data=feature(:,cv.test(i));

        % Training set normalization
        Tmin=min(res_Train_Data,[],2);
        Tmax=max(res_Train_Data,[],2);
        res_Train_Data=(res_Train_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;
        % Test set normalization
        res_Test_Data=(res_Test_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;
        
        %%%%%%%%%%%%%%%%%%%%%%% Polynomial Expansion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        TrainData=gpuArray(Taylor_expan(res_Train_Data,Power));
        TestData=gpuArray(Taylor_expan(res_Test_Data,Power));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        TrainLabels = gpuArray(onehot(:,cv.training(i)));
        TrainLabels_valid=gpuArray(Labels(:,cv.training(i)));
        TestLabels=gpuArray(Labels(:,cv.test(i)));
    
        TrainData_valid=TrainData; % For training set validation
        res_Train_Data_valid=res_Train_Data;

        % Initialize weight matrices
        % 1Random initialization, 2Gaussian initialization, 3Xavier-logistic initialization, 4Xavier-tanh initialization, 5Random initialization
        W_1 = gpuArray(Initialization(hidden_size, expan_size,2)); % Polynomial layer-Res layer weight initialization
        W_2 = gpuArray(Initialization(output_size, hidden_size,2)); % Fault indicator-Voting layer weight initialization
        W_3 = gpuArray(Initialization(output_size, output_size,2)); % Voting-Voting analysis layer weight initialization
        
        train_accurancy=0;
        test_accurancy=0;

        switch Gaus_type
            case 1
                gamma=0;
                beta=0;
            case 2
                gamma=0;
                beta=0;
            case 3
                gamma=ones(hidden_size,1);
                beta=zeros(hidden_size,1);
            case 4
                gamma=ones(hidden_size,1);
                beta=zeros(hidden_size,1);
            case 5
                gamma=0;
                beta=0;
        end
        
        %Initialize gradient mean (first moment):
        GW1=zeros(size(W_1));
        GW2=zeros(size(W_2)); 
        GW3=zeros(size(W_3));
        GWgamma=zeros(size(gamma));
        GWbeta=zeros(size(beta));
        %Initialize gradient variance without mean subtraction (second moment):
        MW1=zeros(size(W_1));
        MW2=zeros(size(W_2)); 
        MW3=zeros(size(W_3));
        MWgamma=zeros(size(gamma));
        MWbeta=zeros(size(beta));

%         % Calculate mean and standard deviation
%         uk = mean(res_Train_Data(:,1:486),2); % Change res_data
%         sigma = std(res_Train_Data(:,1:486),0,2);
%         ub = uk + 3*sigma;
%         if i == 1 && testingNum == 1
%             UB = zeros(10,numEpochs);
%         end

        if testingNum == 1 && i == 1
            W1=zeros([size(W_1) numEpochs]); % Calculate cumulative gradients for feature importance visualization
            W2=zeros([size(W_2) numEpochs]); 
            W3=zeros([size(W_3) numEpochs]);
        end

        % Train network
        for epoch = 1:numEpochs
            % Training on training set
            input_data = TrainData;
            res_data = res_Train_Data;
            %%%%%%%%%%%%% Forward Propagation %%%%%%%%%%%%%

            X1 = W_1*input_data;
            X1_acti = Activate(X1,Acti_type_1); %Activate is the activation function

            % Calculate difference
            et = Edifference(res_data,X1_acti,diff_type);

            % Transform to normal distribution
            [et_Gau, normalized_input, mean_val, var_val] = Gaussian_trans(et,gamma,beta,Gaus_type); % 1None, 2Log transform, 3BN, 4LN


            % Calculate mean and standard deviation
            uk = mean(et_Gau,2); % Change res_data
            sigma = std(et_Gau,0,2);
            ub = uk + 3*sigma; % Get upper bound
            ub = ub * 0.9; % Adjust boundary size to not exceed 1


            st = 3-(4./(1+exp(-et_Gau*log(3)./(ub))));  % Get voting value

            X2 = W_2 * st;
            X2_acti = Activate(X2,Acti_type_2);
            
            X3 = W_3 * X2_acti;

            X3_acti = -1 * X3;
            output = softmax(X3_acti);

            %%%%%%%%%%%% Backward Propagation %%%%%%%%%%%%%%
            d_Loss = output - TrainLabels;
            
            d_X3_acti = d_Loss;

            d_X3 = -1 * d_X3_acti;
            
            d_W3 = d_X3 * X2_acti' / train_num;
            d_X2_acti = W_3' * d_X3;

            d_X2 = d_X2_acti .* Activate_grad(X2,Acti_type_2);

            d_W2 = d_X2 * st' / train_num;
            d_st = W_2' * d_X2;
            
            % Formula for s_t
            d_et_Gau = d_st .* -(4 * exp(-et * log(3) ./ (ub)) * log(3) ./ (ub)) ./ (1 + exp(-et * log(3) ./ (ub))).^2;
            % d_et_Gau = d_st;
            [d_et,d_gamma,d_beta] = Gaussian_trans_grad(d_et_Gau,et,normalized_input,gamma,mean_val,var_val,Gaus_type);

            d_x1_acti = Edifference_grad(res_Train_Data,X1_acti,diff_type,d_et);

            d_x1 = d_x1_acti .* Activate_grad(X1,Acti_type_1);


            % For W_PR
            d_W1 = d_x1 * TrainData' / train_num;
            
            % Update parameters
            [W_1,GW1,MW1]=Gradient_renewal(7,W_1,d_W1,GW1,MW1,lr,epoch); % Use learning rate decay and gradient direction optimization: 1Adjustable decay, 2AdaGrad,
            [W_2,GW2,MW2]=Gradient_renewal(7,W_2,d_W2,GW2,MW2,lr,epoch); %     3RMSprop, 4Momentum, 5AdaM, 6AdaM_2, 7AdaM_3
            [W_3,GW3,MW3]=Gradient_renewal(7,W_3,d_W3,GW3,MW3,lr,epoch);

            gamma = gamma - lr * d_gamma;
            beta = beta - lr * d_beta;

            if testingNum == 1 && i == 1
                W1(:, :, epoch) = d_W1;
                W2(:, :, epoch)  = d_W2;
                W3(:, :, epoch)  = d_W3;
            end


            %% Test on training set
            output=forward_propagation(TrainData_valid,res_Train_Data_valid,ub,W_1,W_2,W_3,Acti_type_1,Acti_type_2,gamma,beta,diff_type,Gaus_type);
            
            % Calculate accuracy and loss            
            [~, predictedLabels] = max(output);
            judgeLabels = predictedLabels == TrainLabels_valid;
            train_accurancy = mean(judgeLabels);

            %% Test on test set
            output=forward_propagation(TestData,res_Test_Data,ub,W_1,W_2,W_3,Acti_type_1,Acti_type_2,gamma,beta,diff_type,Gaus_type);
            
            % Calculate accuracy and loss            
            [~, predictedLabels] = max(output);
            judgeLabels = predictedLabels == TestLabels;
            test_accurancy = mean(judgeLabels);

            fprintf('No.%d, K=%d, epoch=%d, Train accurancy:%.4f, Test accurancy:%.4f\n',testingNum,i,epoch,train_accurancy*100,test_accurancy*100);

            if i==1 && testingNum==1
                addpoints(h1, epoch, train_accurancy); % Add training set accuracy point to graph
                addpoints(h2, epoch, test_accurancy); % Add test set accuracy point to graph
            end
            drawnow; % Update figure window
        end

        % Calculate accuracy for each class on test set
        for class_number = 1:class_num
            class_acc(class_number) = class_acc(class_number) + mean(judgeLabels(1+(class_number-1)*54:class_number*54));  % Where 54 is the number of samples per class in test set
        end

        accuracy(i)=test_accurancy;
        confusionMatrix(:,:,i) = confusionmat(TestLabels, predictedLabels);

    end
    Total_acc=Total_acc+mean(accuracy);
    Total_conf_acc=Total_conf_acc+mean(confusionMatrix,3);
end
time=toc(t0);

Average_accuracy=Total_acc*100/testingNum;

class_acc = class_acc/(k*testNum);


% 
% %Output results
% fprintf('\n\t Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, Average time: %.4f\n', accuracy, sensitivity, specificity,time/(testNum*k));
fprintf('\n\t Accuracy: %.4f, Average time: %.4f\n', Average_accuracy, time/(testNum*k));
for class_number = 1:class_num
    fprintf('Class %d: %.4f\n', class_number, class_acc(class_number));
end


Total_conf_acc=Total_conf_acc/(testNum*54);

% Initialize precision, recall and F1score arrays
precision = zeros(1, class_num);
recall = zeros(1, class_num);
F1score = zeros(1, class_num);

% Calculate accuracy
accuracy = trace(Total_conf_acc) / sum(Total_conf_acc, 'all');

% Calculate precision, recall and F1score for each class
for i = 1:class_num
    tp = Total_conf_acc(i, i);  % True positive
    fp = sum(Total_conf_acc(:, i)) - tp;  % False positive
    fn = sum(Total_conf_acc(i, :)) - tp;  % False negative
    precision(i) = tp / (tp + fp);  % Precision
    recall(i) = tp / (tp + fn);  % Recall
    F1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));  % F1 score
end

fprintf('Precision: %.4f\n', mean(precision));
fprintf('Recall: %.4f\n', mean(recall));
fprintf('F1 Score: %.4f\n', mean(F1score));

% plotCategoryBoxplots(gather(st), 16, [10 10 800 500])

%% Feature importance visualization
% Calculate cumulative gradients for feature importance measurement
W1_accu = zeros(size(W_1));
W2_accu = zeros(size(W_2));
W3_accu = zeros(size(W_3));
% for i = 1:numEpochs
%     W1_accu = (W1_accu*0.5 + W1(:, :, 1)*0.5) * 0.999^(numEpochs+1-i);
%     W2_accu = (W2_accu*0.5 + W2(:, :, 1)*0.5) * 0.999^(numEpochs+1-i);
%     W3_accu = (W3_accu*0.5 + W3(:, :, 1)*0.5) * 0.999^(numEpochs+1-i);
% end
for i = 1:numEpochs
    W1_accu = W1_accu + W1(:, :, 1) * 0.99^(numEpochs+1-i);
    W2_accu = W2_accu + W2(:, :, 1) * 0.99^(numEpochs+1-i);
    W3_accu = W3_accu + W3(:, :, 1) * 0.99^(numEpochs+1-i);
end
W1_accu = abs(mean(W1,3)); % Normalization
W2_accu = abs(mean(W2,3));
W3_accu = abs(mean(W3,3));


figure;
imagesc(W1(:, :, 800));
colorbar;
title('W1: epoch 10', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(W1(:, :, 900));
colorbar;
title('W1: epoch 100', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(W1(:, :, 1000));
colorbar;
title('W1: epoch 1000', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(abs(W1_accu));
colorbar;
title('W1', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');

figure;
imagesc(abs(W2(:, :, 10)));
colorbar;
title('W2: epoch 10', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(abs(W2(:, :, 100)));
colorbar;
title('W2: epoch 100', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(abs(W2(:, :, 1000)));
colorbar;
title('W2: epoch 1000', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');
figure;
imagesc(abs(W2_accu));
colorbar;
title('W2', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');

figure;
imagesc(W3_accu);
colorbar;
title('W3', 'FontName', 'Times New Roman', 'FontSize', 16);
set(gca, 'XTick', [], 'YTick', []);
colormap('parula');

figure;
imagesc(W3_accu*W2_accu);
colorbar;
% Set title
title('Correlation Matrix', 'FontName', 'Times New Roman', 'FontSize', 16);
% Turn off axis ticks
set(gca, 'XTick', 1:13, 'YTick', 1:14);
% Set x and y axis labels
xlabel('Feature Types', 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Fault Types', 'FontName', 'Times New Roman', 'FontSize', 14);
% Define x and y axis labels
xticklabels({'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13'});
yticklabels({'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'N'});
% Set colormap
colormap('parula');