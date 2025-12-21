%% EECE5644 - Assignment 4 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 1
% Tasks: Train and test Support Vector Machine (SVM) and Multi-layer Perceptron (MLP) classifiers that aim for minimum probability of classification error

clear all; close all; clc;

%% Part A: SVM with K-fold Cross-Validation
% Adapted from: svmKfoldCrossValidation.m

% Generate Data
fprintf('Question 1: Concentric Circles Classification \n\n');
fprintf(' ---------- Part A: SVM Classification ---------- \n\n');

fprintf('Step 1: Generating Data: \n');

% Parameters as specified in question
N_train = 1000;
N_test = 10000;
r_neg = 2;          % radius for class -1
r_pos = 4;          % radius for class +1
sigma = 1;          % noise standard deviation

% Generate training data
[x_train, labels_train] = generateConcentricCirclesData(N_train, r_neg, r_pos, sigma);
fprintf('Training data: %d samples\n', N_train);

% Generate test data (independent samples)
[x_test, labels_test] = generateConcentricCirclesData(N_test, r_neg, r_pos, sigma);
fprintf('Test data: %d samples\n', N_test);

% Visualize the training data
figure(1); clf;
plot(x_train(1, labels_train==-1), x_train(2, labels_train==-1), 'ro', 'MarkerSize', 4);
hold on;
plot(x_train(1, labels_train==1), x_train(2, labels_train==1), 'b+', 'MarkerSize', 4);
axis equal; grid on;
xlabel('x_1'); ylabel('x_2');
title('Training Data: Concentric Circles');
legend('Class -1 (inner)', 'Class +1 (outer)');

% SVM with K-fold Cross-Validation
% Adapted code from svmKfoldCrossValidation.m

fprintf('\nStep 2: SVM Training with K-fold Cross-Validation: \n');

K = 10;         % 10-fold cross-validation

% Defining hyperparameter search space
% Box Constraint (C) values and Kernel Scale (sigma) values  
CList = 10.^linspace(-2, 3, 12);  % [0.01, 0.0215, ..., 1000]
sigmaList = 10.^linspace(-1, 1.5, 12);  % [0.1, 0.162, ..., 31.6]

fprintf('Testing %d C values: [%.4f, ..., %.2f]\n', length(CList), CList(1), CList(end));
fprintf('Testing %d sigma values: [%.4f, ..., %.2f]\n', length(sigmaList), sigmaList(1), sigmaList(end));
fprintf('Total combinations to test: %d\n', length(CList) * length(sigmaList));

% Prepare K-fold partitions 
% Adapted from svmKfoldCrossValidation.m
dummy = ceil(linspace(0, N_train, K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1, dummy(k+1)];
end

% Initialize results matrix for faster processing
PCorrect = zeros(length(CList), length(sigmaList));

% Cross-validation loop (structure from svmKfoldCrossValidation.m)
fprintf('\nRunning cross-validation -\n');
for sigmaCounter = 1:length(sigmaList)
    fprintf('  Sigma %d/%d (%.3f)\n', sigmaCounter, length(sigmaList), sigmaList(sigmaCounter));
    sigma_val = sigmaList(sigmaCounter);
    
    for CCounter = 1:length(CList)
        C_val = CList(CCounter);
        
        % K-fold validation
        Ncorrect = zeros(K, 1);
        
        for k = 1:K
            % Split data into train and validation folds
            indValidate = indPartitionLimits(k,1):indPartitionLimits(k,2);
            x_validate = x_train(:, indValidate);
            labels_validate = labels_train(indValidate);
            
            if k == 1
                indTrain = indPartitionLimits(k,2)+1:N_train;
            elseif k == K
                indTrain = 1:indPartitionLimits(k,1)-1;
            else
                indTrain = [1:indPartitionLimits(k,1)-1, indPartitionLimits(k,2)+1:N_train];
            end
            
            x_train_fold = x_train(:, indTrain);
            labels_train_fold = labels_train(indTrain);
            
            % Train SVM (adapted from svmDemo.m and svmKfoldCrossValidation.m)
            SVMk = fitcsvm(x_train_fold', labels_train_fold, 'BoxConstraint', C_val, 'KernelFunction', 'gaussian', 'KernelScale', sigma_val);
            
            % Validate
            decisions_validate = SVMk.predict(x_validate')';
            Ncorrect(k) = sum(labels_validate == decisions_validate);
        end
        
        % Average accuracy across folds
        PCorrect(CCounter, sigmaCounter) = sum(Ncorrect) / N_train;
    end
end
fprintf('Cross-validation complete!\n\n');

% Finding the best hyperparameters
[maxAccuracy, maxIdx] = max(PCorrect(:));
[indBestC, indBestSigma] = ind2sub(size(PCorrect), maxIdx);
bestC = CList(indBestC);
bestSigma = sigmaList(indBestSigma);

fprintf('Step 3: Best Hyperparameters Found\n');
fprintf('  Best C: %.4f\n', bestC);
fprintf('  Best Sigma: %.4f\n', bestSigma);
fprintf('  Cross-validation accuracy: %.4f (%.2f%%)\n', maxAccuracy, maxAccuracy*100);
fprintf('  Cross-validation error: %.4f (%.2f%%)\n\n', 1-maxAccuracy, (1-maxAccuracy)*100);

% Visualizing cross-validation results
figure(2); clf;

% For each sigma value, plotting Accuracy vs C
subplot(1,2,1);
for sigmaCounter = 1:length(sigmaList)
    plot(log10(CList), PCorrect(:, sigmaCounter), 'o-', 'LineWidth', 1.5, 'DisplayName', sprintf('\\sigma=%.2f', sigmaList(sigmaCounter)));
    hold on;
end
xlabel('log_{10}(C)'); 
ylabel('Cross-Validation Accuracy');
title('SVM Cross-Validation Accuracy vs C');
grid on;
legend('Location', 'best');
% Marking the best point
plot(log10(bestC), maxAccuracy, 'r*', 'MarkerSize', 15, 'LineWidth', 3);

% For each C value, plot Accuracy vs Sigma
subplot(1,2,2);
for CCounter = 1:length(CList)
    plot(log10(sigmaList), PCorrect(CCounter, :), 'o-', 'LineWidth', 1.5, 'DisplayName', sprintf('C=%.2f', CList(CCounter)));
    hold on;
end
xlabel('log_{10}(Sigma)'); 
ylabel('Cross-Validation Accuracy');
title('SVM Cross-Validation Accuracy vs Sigma');
grid on;
legend('Location', 'best');
% Mark best point
plot(log10(bestSigma), maxAccuracy, 'r*', 'MarkerSize', 15, 'LineWidth', 3);

% Training final SVM on all training data
fprintf('Step 4: Training Final SVM on All Training Data \n');
SVMBest = fitcsvm(x_train', labels_train, 'BoxConstraint', bestC, 'KernelFunction', 'gaussian', 'KernelScale', bestSigma);
fprintf('Training complete!\n\n');

% Testing on test set
fprintf('Step 5: Testing on Test Set: \n');
decisions_test = SVMBest.predict(x_test')';

% Calculate test error 
indCorrect = find(labels_test == decisions_test);
indIncorrect = find(labels_test ~= decisions_test);
testAccuracy = length(indCorrect) / N_test;
testError = 1 - testAccuracy;

fprintf('Test Results:\n');
fprintf('  Correct classifications: %d / %d\n', length(indCorrect), N_test);
fprintf('  Test Accuracy: %.4f (%.2f%%)\n', testAccuracy, testAccuracy*100);
fprintf('  Test Error: %.4f (%.2f%%)\n\n', testError, testError*100);

% Visualize test performance with custom colors
fprintf('Step 6: Visualizing Test Performance: \n');
figure(3); clf;

% Create logical arrays for correct/incorrect classifications
isCorrect = (labels_test == decisions_test);
isIncorrect = ~isCorrect;
isClass_neg1 = (labels_test == -1);
isClass_pos1 = (labels_test == 1);

% Plot test data with custom colors:
% Inner circle (class -1) = GREEN;  Outer circle (class +1) = BLUE;  Incorrect = RED
plot(x_test(1, isCorrect & isClass_neg1), x_test(2, isCorrect & isClass_neg1), 'go', 'MarkerSize', 3, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'none', 'DisplayName', 'Class -1 Correct (Inner)');
hold on;
plot(x_test(1, isCorrect & isClass_pos1), x_test(2, isCorrect & isClass_pos1), 'b+', 'MarkerSize', 3, 'LineWidth', 0.5, 'DisplayName', 'Class +1 Correct (Outer)');
plot(x_test(1, isIncorrect), x_test(2, isIncorrect), 'rx', 'MarkerSize', 6, 'LineWidth', 0.5, 'DisplayName', 'Incorrect');

% Create finer grid for smoother boundary
Nx = 300; Ny = 300;
xGrid = linspace(-8, 8, Nx);
yGrid = linspace(-8, 8, Ny);
[h, v] = meshgrid(xGrid, yGrid);

fprintf('Computing decision boundary on %dx%d grid- \n', Nx, Ny);
decisionGrid = SVMBest.predict([h(:), v(:)]);
zGrid = reshape(decisionGrid, Ny, Nx);

% Plot decision boundary
contour(xGrid, yGrid, zGrid, [0, 0], 'k-', 'LineWidth', 3, 'DisplayName', 'Decision Boundary');

% Add reference circles at r=2 and r=4 for comparison
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'c--', 'LineWidth', 1.5, 'DisplayName', 'True r=2 circle');
plot(4*cos(theta), 4*sin(theta), 'c--', 'LineWidth', 1.5, 'DisplayName', 'True r=4 circle');

axis equal; 
xlim([-8 8]); ylim([-8 8]);
grid on;
xlabel('x_1'); ylabel('x_2');
title(sprintf('SVM Test Performance: Accuracy=%.2f%%, Error=%.2f%%', ...
              testAccuracy*100, testError*100));
legend('Location', 'best');

fprintf('SVM Analysis Complete!\n\n');

%%
% Confusion Matrix for SVM
fprintf('\n\nStep 5b: SVM Confusion Matrix: \n');

% Convert to -1/+1 for clarity
true_neg1 = sum((labels_test == -1) & (decisions_test == -1));
false_pos1 = sum((labels_test == -1) & (decisions_test == 1));
false_neg1 = sum((labels_test == 1) & (decisions_test == -1));
true_pos1 = sum((labels_test == 1) & (decisions_test == 1));

confusion_svm = [true_neg1, false_pos1; false_neg1, true_pos1];

fprintf('Confusion Matrix (rows=actual, cols=predicted):\n');
fprintf('              Pred -1    Pred +1\n');
fprintf('Actual -1:    %6d     %6d\n', true_neg1, false_pos1);
fprintf('Actual +1:    %6d     %6d\n\n', false_neg1, true_pos1);

% Additional metrics
sensitivity = true_pos1 / (true_pos1 + false_neg1);  % Recall for +1 class
specificity = true_neg1 / (true_neg1 + false_pos1);  % Recall for -1 class
precision_pos = true_pos1 / (true_pos1 + false_pos1);
precision_neg = true_neg1 / (true_neg1 + false_neg1);

fprintf('Additional Metrics:\n');
fprintf('  Sensitivity (Recall +1): %.4f\n', sensitivity);
fprintf('  Specificity (Recall -1): %.4f\n', specificity);
fprintf('  Precision (+1): %.4f\n', precision_pos);
fprintf('  Precision (-1): %.4f\n\n', precision_neg);

% Visualize confusion matrix
figure(3.5); clf;
confusionchart(confusion_svm, {'-1 (Inner)', '+1 (Outer)'});
title('SVM Confusion Matrix on Test Set');

%% Part B: MLP with K-fold Cross-Validation
% Multi-layer Perceptron with single hidden layer
% Adapted from: mleMLPwAWGN.m, kfoldMLP.m

fprintf('\n ---------- Part B: MLP Classification ---------- \n\n');

% MLP needs labels as positive integers for classification
% Converting the labels from {-1, +1} to {1, 2}
labels_train_mlp = (labels_train + 1)/2 + 1;    % -1 -> 1 
labels_test_mlp = (labels_test + 1)/2 + 1;      % +1 -> 2

fprintf('Step 1: Data prepared for MLP: \n');
fprintf('  Training samples: %d\n', N_train);
fprintf('  Test samples: %d\n', N_test);
fprintf('  Input dimensions: %d\n', size(x_train, 1));
fprintf('  Output classes: %d\n\n', length(unique(labels_train_mlp)));

% MLP K-fold Cross-Validation
% Hyperparameter: Number of perceptrons in hidden layer
% Adapted from: kfoldMLP.m

fprintf('Step 2: MLP Training with K-fold Cross-Validation: \n');

K = 10;        % Same K-fold as SVM for fair comparison

% Defining hidden layer size search space
nPerceptronsList = [2, 5, 10, 15, 20, 30, 50, 75, 100];
fprintf('Testing %d hidden layer sizes: [%d, ..., %d]\n', length(nPerceptronsList), nPerceptronsList(1), nPerceptronsList(end));

% Using same partition limits as SVM- indPartitionLimits is already defined from SVM part

% Initialize results
PCorrect_MLP = zeros(length(nPerceptronsList), 1);
PError_MLP = zeros(length(nPerceptronsList), 1);

fprintf('\nRunning cross-validation-\n');

% Cross-validation loop (structure from kfoldMLP.m)
for M_counter = 1:length(nPerceptronsList)
    M = nPerceptronsList(M_counter);
    fprintf('  Testing M=%d hidden neurons (%d/%d)\n', M, M_counter, length(nPerceptronsList));
    
    Ncorrect = zeros(K, 1);
    
    for k = 1:K
        % Split data (same as SVM)
        indValidate = indPartitionLimits(k,1):indPartitionLimits(k,2);
        x_validate = x_train(:, indValidate);
        labels_validate = labels_train_mlp(indValidate);
        
        if k == 1
            indTrain = indPartitionLimits(k,2)+1:N_train;
        elseif k == K
            indTrain = 1:indPartitionLimits(k,1)-1;
        else
            indTrain = [1:indPartitionLimits(k,1)-1, indPartitionLimits(k,2)+1:N_train];
        end
        
        x_train_fold = x_train(:, indTrain);
        labels_train_fold = labels_train_mlp(indTrain);
        
        % Create MLP network (adapted from mleMLPwAWGN.m)
        % Single hidden layer with M perceptrons
        net = patternnet(M);
        
        % Configure network
        net.trainParam.showWindow = false;  % Don't show training GUI
        net.trainParam.epochs = 1000;       % Maximum epochs - can be increased to 2000 if needed
        net.trainParam.goal = 1e-5;         % Performance goal
        
        % Set activation function for hidden layer
        % Using 'tansig' (hyperbolic tangent sigmoid) as default. Could also try 'logsig' (logistic sigmoid) or 'radbas' (radial basis)
        net.layers{1}.transferFcn = 'radbas';
        
        % Output layer uses softmax by default for pattern recognition
        
        % Prepare data in format MLP expects (targets as one-hot vectors)
        targets_train = zeros(2, length(labels_train_fold));
        for i = 1:length(labels_train_fold)
            targets_train(labels_train_fold(i), i) = 1;
        end
        
        % Train network (adapted from mleMLPwAWGN.m)
        net = train(net, x_train_fold, targets_train);
        
        % Validate
        outputs_validate = net(x_validate);
        [~, predictions_validate] = max(outputs_validate, [], 1);
        Ncorrect(k) = sum(predictions_validate == labels_validate);
    end
    
    % Average accuracy across folds
    PCorrect_MLP(M_counter) = sum(Ncorrect) / N_train;
    PError_MLP(M_counter) = 1 - PCorrect_MLP(M_counter);
end

fprintf('Cross-validation complete!\n\n');

% Finding best number of perceptrons
[maxAccuracy_MLP, bestM_idx] = max(PCorrect_MLP);
bestM = nPerceptronsList(bestM_idx);
bestError_MLP = 1 - maxAccuracy_MLP;

fprintf('Step 3: Best Hyperparameter Found\n');
fprintf('  Best M (hidden neurons): %d\n', bestM);
fprintf('  Cross-validation accuracy: %.4f (%.2f%%)\n', maxAccuracy_MLP, maxAccuracy_MLP*100);
fprintf('  Cross-validation error: %.4f (%.2f%%)\n\n', bestError_MLP, bestError_MLP*100);

% Visualize cross-validation results
figure(4); clf;

subplot(1,2,1);
plot(nPerceptronsList, PCorrect_MLP, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(bestM, maxAccuracy_MLP, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('Number of Hidden Neurons (M)');
ylabel('Cross-Validation Accuracy');
title('MLP Cross-Validation: Accuracy vs Hidden Layer Size');
grid on;
legend('CV Accuracy', 'Best M', 'Location', 'best');

subplot(1,2,2);
plot(nPerceptronsList, PError_MLP, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(bestM, bestError_MLP, 'k*', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('Number of Hidden Neurons (M)');
ylabel('Cross-Validation Error');
title('MLP Cross-Validation: Error vs Hidden Layer Size');
grid on;
legend('CV Error', 'Best M', 'Location', 'best');

% Train final MLP on all training data
fprintf('Step 4: Training Final MLP on All Training Data: \n');

net_final = patternnet(bestM);
net_final.trainParam.showWindow = false;        % GUI Window - off
net_final.trainParam.epochs = 1000;             % Epochs
net_final.trainParam.goal = 1e-5;               % Goal
net_final.layers{1}.transferFcn = 'radbas';

% Prepare all training data
targets_train_all = zeros(2, N_train);
for i = 1:N_train
    targets_train_all(labels_train_mlp(i), i) = 1;
end

net_final = train(net_final, x_train, targets_train_all);
fprintf('Training complete!\n\n');

% Testing on test set
fprintf('Step 5: Testing on Test Set: \n');

outputs_test = net_final(x_test);
[~, predictions_test] = max(outputs_test, [], 1);

% Calculate test error
indCorrect_MLP = find(predictions_test == labels_test_mlp);
indIncorrect_MLP = find(predictions_test ~= labels_test_mlp);
testAccuracy_MLP = length(indCorrect_MLP) / N_test;
testError_MLP = 1 - testAccuracy_MLP;

fprintf('Test Results:\n');
fprintf('  Correct classifications: %d / %d\n', length(indCorrect_MLP), N_test);
fprintf('  Test Accuracy: %.4f (%.2f%%)\n', testAccuracy_MLP, testAccuracy_MLP*100);
fprintf('  Test Error: %.4f (%.2f%%)\n\n', testError_MLP, testError_MLP*100);

% Visualize test performance with custom colors
fprintf('Step 6: Visualizing Test Performance: \n');

figure(5); clf;

% Create logical arrays for correct/incorrect classifications
isCorrect_MLP = (predictions_test == labels_test_mlp);
isIncorrect_MLP = ~isCorrect_MLP;
isClass1 = (labels_test_mlp == 1);  % Class -1 in original labels (inner)
isClass2 = (labels_test_mlp == 2);  % Class +1 in original labels (outer)

% Plot test data with custom colors:
% Inner circle (class 1/-1) = GREEN
% Outer circle (class 2/+1) = BLUE
% Incorrect = RED
plot(x_test(1, isCorrect_MLP & isClass1), x_test(2, isCorrect_MLP & isClass1), 'go', 'MarkerSize', 3, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'none', 'DisplayName', 'Class -1 Correct (Inner)');
hold on;
plot(x_test(1, isCorrect_MLP & isClass2), x_test(2, isCorrect_MLP & isClass2), 'b+', 'MarkerSize', 3, 'LineWidth', 0.5, 'DisplayName', 'Class +1 Correct (Outer)');
plot(x_test(1, isIncorrect_MLP), x_test(2, isIncorrect_MLP), 'rx', 'MarkerSize', 6, 'LineWidth', 1, 'DisplayName', 'Incorrect');

% Create finer grid for smoother boundary
Nx = 300; Ny = 300;
xGrid = linspace(-7, 7, Nx);
yGrid = linspace(-7, 7, Ny);
[h, v] = meshgrid(xGrid, yGrid);
gridPoints = [h(:), v(:)]';

fprintf('Computing decision boundary on %dx%d grid- \n', Nx, Ny);
outputs_grid = net_final(gridPoints);
[~, decisions_grid] = max(outputs_grid, [], 1);
zGrid = reshape(decisions_grid, Ny, Nx);

% Plot decision boundary
contour(xGrid, yGrid, zGrid, [1.5 1.5], 'k-', 'LineWidth', 3, 'DisplayName', 'Decision Boundary');

% Add reference circles at r=2 and r=4 for comparison
theta = linspace(0, 2*pi, 100);
plot(2*cos(theta), 2*sin(theta), 'c--', 'LineWidth', 1.5, 'DisplayName', 'True r=2 circle');
plot(4*cos(theta), 4*sin(theta), 'c--', 'LineWidth', 1.5, 'DisplayName', 'True r=4 circle');

axis equal; 
xlim([-8 8]); ylim([-8 8]);
grid on;
xlabel('x_1'); ylabel('x_2');
title(sprintf('MLP Test Performance: Accuracy=%.2f%%, Error=%.2f%%', testAccuracy_MLP*100, testError_MLP*100));
legend('Location', 'best');

fprintf('MLP Analysis Complete!\n\n');

%% Compare SVM vs MLP
fprintf('---------- COMPARISON: SVM vs MLP ---------- \n\n');
fprintf('Cross-Validation Results:\n');
fprintf('  SVM  - Best C=%.4f, Sigma=%.4f, CV Error=%.2f%%\n', bestC, bestSigma, (1-maxAccuracy)*100);
fprintf('  MLP  - Best M=%d, CV Error=%.2f%%\n', bestM, bestError_MLP*100);
fprintf('\nTest Set Results:\n');
fprintf('  SVM  - Test Error: %.2f%%\n', testError*100);
fprintf('  MLP  - Test Error: %.2f%%\n', testError_MLP*100);
fprintf('\n');

% Summary comparison plot
figure(6); clf;
models = categorical({'SVM', 'MLP'});
cv_errors = [(1-maxAccuracy)*100, bestError_MLP*100];
test_errors = [testError*100, testError_MLP*100];

subplot(1,2,1);
bar(models, cv_errors);
ylabel('Error Rate (%)');
title('Cross-Validation Error Comparison');
grid on;
ylim([0 max([cv_errors, test_errors])*1.2]);

subplot(1,2,2);
bar(models, test_errors);
ylabel('Error Rate (%)');
title('Test Error Comparison');
grid on;
ylim([0 max([cv_errors, test_errors])*1.2]);

fprintf('Question 1 Complete!\n');