%% 데이터 저장 과정
%regressionLearner 도움 얻기
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'COUNT'};
%varName = {'N02';'O3';'CO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'PEOPLE'};
%varName = {'N02';'O3';'CO2';'SO2';'PM10';'PM2.5';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'YEAR';'PEOPLE'};
%varName = {'N02';'O3';'CO2';'SO2';'PM10';'PM2.5';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'O3';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
varName = {'N02';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};
%varName = {'N02';'CO2';'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'Culture';'PEOPLE'};
%varName ={'SO2';'PM10';'HOLIDAY';'SEASON';'TEMPERATURE';'HUMIDITY';'WIND';'PEOPLE'};성능이 제일 좋음
tbl.Properties.VariableNames = varName;

%% NaN 찾기 -> 누락된 데이터 찾기
M = table2array(tbl);
any(isnan(M(:,:)))

%% 데이터 섞기 
[r c] = size(tbl);
tbl = tbl(randperm(r),:);
%% 상관계수 구하기
[coef p] = corrcoef(M); 
coef = tril(coef);
h = heatmap(coef);
h.XData = varName;
h.YData = varName; 
h.FontName = 'Cambria';
h.Colormap = pink;
%% 유의확률(P-Value 구하기)
p= tril(p);
h = heatmap(p);
h.XData = varName;
h.YData = varName; 
h.FontName = 'Cambria';
h.Colormap = pink;

%% Train Data Test Data 나누기
[r c] = size(tbl);
idx = int16(r*(8/10));
train_x = tbl(1:idx,1:end-1);
test_x =  tbl(idx+1:end,1:end-1);

train_y = tbl(1:idx,end);
test_y = tbl(idx+1:end,end);

%% SVM 어떤 커널로 학습을 진행할 지 택하기
%Linear Kernel로 학습 진행 kFold Validation을 k=5으로 진행
MdlLin = fitrsvm(tbl,'PEOPLE','Standardize',true,'KFold',5,'KernelScale','auto');
%Gaussian Kernel로 학습 진행 kFold Validation을 k=5으로 진행
MdlGau = fitrsvm(tbl,'PEOPLE','Standardize',true,'KFold',5,'KernelFunction','gaussian','KernelScale','auto');
%% 학습 시키기
MdlLin.Trained
MdlGau.Trained

%% 훈련 결과 수치로 나타내기 -> 성능 좋은 것 택하기 (오차가 적은 커널 선택)
mseLin = kfoldLoss(MdlLin)
mseGau = kfoldLoss(MdlGau)
%% 더 좋은 것으로 선택
%성능 좋은것으로 실행
%Mdl = fitrsvm(train_x,train_y,'KernelFunction','gaussian','KernelScale','auto','Standardize',true',...
%   'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'));
Mdl = fitrsvm(train_x,train_y,'KernelFunction','gaussian','KernelScale','auto','Standardize',true');

% 수렴했는지 확인 0이면 다시 학습
Mdl.ConvergenceInfo.Converged
%학습 횟수 출력
iter = Mdl.NumIterations

%% Predict
pre_y = predict(Mdl,test_x);
test_y = table2array(test_y);

ty = table2array(train_y);
py = predict(Mdl,train_x);
%% 결과 보여주기 - 그래프 + LossFunction
%그래프로 나타내기
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
