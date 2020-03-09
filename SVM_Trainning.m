%% ������ ���� ����
%regressionLearner ���� ���
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'COUNT'};
%varName = {'N02';'O3';'CO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'PEOPLE'};
%varName = {'N02';'O3';'CO2';'SO2';'PM10';'PM2.5';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'YEAR';'PEOPLE'};
%varName = {'N02';'O3';'CO2';'SO2';'PM10';'PM2.5';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'O3';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
varName = {'N02';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'Culture';'PEOPLE'};
%varName ={'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};������ ���� ����
tbl.Properties.VariableNames = varName;

%% NaN ã�� -> ������ ������ ã��
M = table2array(tbl);
any(isnan(M(:,:)))

%% ������ ���� 
[r c] = size(tbl);
tbl = tbl(randperm(r),:);
%% ������ ���ϱ�
[coef p] = corrcoef(M); 
coef = tril(coef);
h = heatmap(coef);
h.XData = varName;
h.YData = varName; 
h.FontName = 'Cambria';
h.Colormap = pink;
%% ����Ȯ��(P-Value ���ϱ�)
p= tril(p);
h = heatmap(p);
h.XData = varName;
h.YData = varName; 
h.FontName = 'Cambria';
h.Colormap = pink;

%% Train Data Test Data ������
[r c] = size(tbl);
idx = int16(r*(8/10));
train_x = tbl(1:idx,1:end-1);
test_x =  tbl(idx+1:end,1:end-1);

train_y = tbl(1:idx,end);
test_y = tbl(idx+1:end,end);

%% SVM � Ŀ�η� �н��� ������ �� ���ϱ�
%Linear Kernel�� �н� ���� kFold Validation�� k=5���� ����
MdlLin = fitrsvm(tbl,'PEOPLE','Standardize',true,'KFold',5,'KernelScale','auto');
%Gaussian Kernel�� �н� ���� kFold Validation�� k=5���� ����
MdlGau = fitrsvm(tbl,'PEOPLE','Standardize',true,'KFold',5,'KernelFunction','gaussian','KernelScale','auto');
%% �н� ��Ű��
MdlLin.Trained
MdlGau.Trained

%% �Ʒ� ��� ��ġ�� ��Ÿ���� -> ���� ���� �� ���ϱ� (������ ���� Ŀ�� ����)
mseLin = kfoldLoss(MdlLin)
mseGau = kfoldLoss(MdlGau)
%% �� ���� ������ ����
%���� ���������� ����
%Mdl = fitrsvm(train_x,train_y,'KernelFunction','gaussian','KernelScale','auto','Standardize',true',...
%   'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'));
Mdl = fitrsvm(train_x,train_y,'KernelFunction','gaussian','KernelScale','auto','Standardize',true');

% �����ߴ��� Ȯ�� 0�̸� �ٽ� �н�
Mdl.ConvergenceInfo.Converged
%�н� Ƚ�� ���
iter = Mdl.NumIterations

%% Predict
pre_y = predict(Mdl,test_x);
test_y = table2array(test_y);

ty = table2array(train_y);
py = predict(Mdl,train_x);
%% ��� �����ֱ� - �׷��� + LossFunction
%�׷����� ��Ÿ����
figure(1)
hold on
scatter(1:length(ty),ty,'b','filled');
scatter(1:length(py),py,'r','filled');
hold off

figure(2)
hold on
scatter(1:length(test_y),test_y,'b','filled');
scatter(1:length(pre_y),pre_y,'r','filled');
hold off
sqrt(mean((test_y-pre_y).^2))%RMSE
mapeLoss = 100*mean(abs((test_y-pre_y)./test_y))%MAPE
%lStd = resubLoss(Mdl)
%%
f = @(Ytrain,Ytest,W)...
    100*mean(abs((Ytest-Ytrain)./Ytest));
%%
