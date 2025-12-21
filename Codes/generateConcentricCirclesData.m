%% EECE5644 - Assignment 4 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 1 - Helper Function to generate concentric circle data
% Task: Generates N samples for concentric circles classification problem

function [x, labels] = generateConcentricCirclesData(N, r_neg, r_pos, sigma)
% Adapted from: generateDataFromGMM.m

% Determine number of samples per class (approximately equal)
N_neg = floor(N/2);
N_pos = N - N_neg;

% Generate class -1 (inner circle)
theta_neg = -pi + 2*pi*rand(1, N_neg);  % Uniform[-π, π]
x_neg = r_neg * [cos(theta_neg); sin(theta_neg)];
% Add Gaussian noise 
% Adapted from randGaussian.m approach
x_neg = x_neg + sigma * randn(2, N_neg);
labels_neg = -ones(1, N_neg);

% Generate class +1 (outer circle)
theta_pos = -pi + 2*pi*rand(1, N_pos);  % Uniform[-π, π]
x_pos = r_pos * [cos(theta_pos); sin(theta_pos)];
% Add Gaussian noise
x_pos = x_pos + sigma * randn(2, N_pos);
labels_pos = ones(1, N_pos);

% Combine both classes
x = [x_neg, x_pos];
labels = [labels_neg, labels_pos];

% Shuffle the data
shuffleIdx = randperm(N);
x = x(:, shuffleIdx);
labels = labels(shuffleIdx);

end