%% EECE5644 - Assignment 4 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 1
% Tasks: GMM-based clustering to segment a color image

clear all; close all; clc;

fprintf(' ---------- Question 2: GMM-Based Image Segmentation ---------- \n\n');

%% Step 1: Load Image
fprintf('Step 1: Loading Image: \n');

filename = '299091.jpg';    % Image of choice 

if ~exist(filename, 'file')
    error('Image file not found!');
end

imdata = imread(filename);
fprintf('Image loaded: %s\n', filename);

%% Step 2: Extract and Normalize Features
% Adapted from: ExampleKmeansImageSegmentation.m
fprintf('Step 2: Extracting and normalizing features: \n');

if length(size(imdata)) == 2        % Grayscale -> Discard
    error('Please use a color (RGB) image');
elseif length(size(imdata)) == 3 % color image
    [R, C, D] = size(imdata);
    N = R * C;
    imdata = double(imdata);
    
    % Creating position and color features (same as K-means)
    rowIndices = (1:R)' * ones(1, C);
    colIndices = ones(R, 1) * (1:C);
    
    % Extracting each color channel separately
    red_channel = imdata(:,:,1);
    green_channel = imdata(:,:,2);
    blue_channel = imdata(:,:,3);
    
    % 5D feature vector as per the question requirements
    features = [rowIndices(:)';           % Row positions
                colIndices(:)';           % Column positions
                red_channel(:)';          % Red
                green_channel(:)';        % Green
                blue_channel(:)'];        % Blue
    
    % Normalize each feature to [0,1] (exactly as in K-means example)
    minf = min(features, [], 2);
    maxf = max(features, [], 2);
    ranges = maxf - minf;
    ranges(ranges == 0) = 1;  % Avoid division by zero
    
    x = diag(ranges.^(-1)) * (features - repmat(minf, 1, N));
end

d = size(x, 1);  % Feature dimensionality (should be 5)
% Debugging-Outputs
fprintf('  Features: %d dimensions, %d pixels\n', d, N);
fprintf('  Image size: %d x %d\n\n', R, C);

%% Step 3: K-fold Cross-Validation for Model Order Selection
fprintf('Step 3: K-fold Cross-Validation for model order selection: \n');

K = 5;  % 5-fold cross-validation
M_values = [2, 3, 4, 5, 6];  % Test multiple M values
n_models = length(M_values);

fprintf('  Using K=%d fold cross-validation\n', K);
fprintf('  Testing M = [%s]\n', num2str(M_values));

% Prepare K-fold partitions
dummy = ceil(linspace(0, N, K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1, dummy(k+1)];
end

X_all = x';     % Transpose for fitgmdist

% Storage for CV results
avgLogLikelihood_train = zeros(n_models, 1);
avgLogLikelihood_val = zeros(n_models, 1);

options = statset('MaxIter', 1000, 'Display', 'off');

% Cross-validation loop
for m_idx = 1:n_models
    M = M_values(m_idx);
    fprintf('  Testing M = %d (%d/%d) ', M, m_idx, n_models);
    tic;
    
    fold_loglik_train = zeros(K, 1);
    fold_loglik_val = zeros(K, 1);
    
    for k = 1:K
        % Split data
        indValidate = indPartitionLimits(k,1):indPartitionLimits(k,2);
        indTrain = setdiff(1:N, indValidate);
        
        X_train = X_all(indTrain, :);
        X_val = X_all(indValidate, :);
        
        try
            % Fit GMM using Maximum Likelihood (EM algorithm)
            gmm = fitgmdist(X_train, M, 'Replicates', 3, 'Options', options, 'RegularizationValue', 1e-5);
            
            % Compute log-likelihood per sample
            loglik_train = sum(log(pdf(gmm, X_train)));
            fold_loglik_train(k) = loglik_train / size(X_train, 1);
            
            loglik_val = sum(log(pdf(gmm, X_val)));
            fold_loglik_val(k) = loglik_val / size(X_val, 1);
            
        catch ME
            fold_loglik_train(k) = -Inf;
            fold_loglik_val(k) = -Inf;
        end
    end
    
    % Average across folds (as required by question)
    avgLogLikelihood_train(m_idx) = mean(fold_loglik_train);
    avgLogLikelihood_val(m_idx) = mean(fold_loglik_val);
    
    elapsed = toc;
    fprintf('Done (%.2f sec, Val LL=%.4f)\n', elapsed, avgLogLikelihood_val(m_idx));
end

fprintf('\nCross-validation complete!\n\n');

%% Step 4: Select Best M (Maximum Validation Log-Likelihood)
[max_val_loglik, best_idx] = max(avgLogLikelihood_val);
best_M = M_values(best_idx);

fprintf('Step 4: Model Selection Results: \n');
fprintf('  Best M = %d (Maximum validation log-likelihood)\n', best_M);
fprintf('  Validation log-likelihood: %.4f\n\n', max_val_loglik);

%% Step 5: Train Final GMMs for All M Values (for comparison)
fprintf('Step 5: Training final GMMs for all M values: \n');

gmm_models = cell(n_models, 1);
label_images = cell(n_models, 1);

for k = 1:n_models
    M = M_values(k);
    fprintf('  Training M = %d ', M);
    tic;
    
    try
        % Train on ALL data
        gmm_models{k} = fitgmdist(X_all, M, 'Replicates', 5, 'Options', options, 'RegularizationValue', 1e-5);
        
        % Assign labels using posterior probabilities (as required)
        posterior_probs = posterior(gmm_models{k}, X_all);
        [~, labels] = max(posterior_probs, [], 2);
        
        % Reshape to image
        label_images{k} = reshape(labels, R, C);
        
        fprintf('Done (%.2f sec)\n', toc);
        
    catch ME
        fprintf('FAILED: %s\n', ME.message);
        label_images{k} = zeros(R, C);
    end
end

fprintf('\n');

%% Step 6: Visualize Results (K-means style layout from the example code provided)
fprintf('Step 6: Creating visualizations: \n');

% Main figure - comparison across M values
figure(1); clf;
set(gcf, 'Position', [100, 100, 1400, 800]);

% Show original image in first position
subplot(2, n_models+1, 1);
imshow(uint8(imdata));
title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');

% Show segmentation for each M value
for k = 1:n_models
    M = M_values(k);
    
    % Top row: Grayscale segmentation
    subplot(2, n_models+1, k+1);
    
    % Normalize labels for display (uniform distribution as required)
    label_display = label_images{k};
    if M > 1
        label_display = uint8(255 * (label_display - 1) / (M - 1));
    else
        label_display = zeros(R, C, 'uint8');
    end
    
    imshow(label_display);
    
    % Highlight best M (from CV)
    if k == best_idx
        title(sprintf('M=%d (BEST)\nVal LL=%.2f', M, avgLogLikelihood_val(k)), 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
    else
        title(sprintf('M=%d\nVal LL=%.2f', M, avgLogLikelihood_val(k)), 'FontSize', 10);
    end
    
    % Bottom row: Color-coded segmentation
    subplot(2, n_models+1, n_models+1+k+1);
    label_rgb = label2rgb(label_images{k}, 'jet', 'k');
    imshow(label_rgb);
    title(sprintf('M=%d (Color)', M), 'FontSize', 10);
end

% Add overall title
sgtitle(sprintf('GMM Segmentation: %s (Best M=%d via CV)', filename, best_M), 'FontSize', 14, 'FontWeight', 'bold');


% Cross-Validation Results Plot
figure(2); clf;
plot(M_values, avgLogLikelihood_train, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Training');
hold on;
plot(M_values, avgLogLikelihood_val, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Validation');
plot(best_M, max_val_loglik, 'k*', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', sprintf('Best M=%d', best_M));
xlabel('Number of GMM Components (M)', 'FontSize', 12);
ylabel('Average Log-Likelihood per Sample', 'FontSize', 12);
title('GMM Model Order Selection via K-fold Cross-Validation', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;


% Main Result - Original and Grayscale Segmentation Side-by-Side
fprintf('Creating main result figure: \n');

figure(3); clf;
set(gcf, 'Position', [100, 100, 1200, 600]);

best_labels = label_images{best_idx};

% Left: Original Image
subplot(1,2,1);
imshow(uint8(imdata));
title('Original Image', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(sprintf('%d x %d pixels', R, C), 'FontSize', 11);

% Right: Grayscale Segmentation (uniformly distributed labels)
subplot(1,2,2);

% Uniformly distribute labels between 0 and 255 for good contrast
% (as specifically required by question)
if best_M > 1
    label_grayscale = uint8(255 * (best_labels - 1) / (best_M - 1));
else
    label_grayscale = zeros(R, C, 'uint8');
end

imshow(label_grayscale);
title(sprintf('GMM-Based Segmentation\nM=%d Components', best_M), 'FontSize', 14, 'FontWeight', 'bold');

% Add colorbar showing component mapping
colormap(gca, gray);
c = colorbar;
c.Ticks = linspace(0, 1, best_M);
c.TickLabels = cellstr(num2str((1:best_M)'));
c.Label.String = 'Component ID';
c.Label.FontSize = 11;

xlabel(sprintf('Labels uniformly distributed: [0, 255]'), 'FontSize', 11);
sgtitle(sprintf('Image Segmentation Result: %s (Selected via %d-fold CV)', filename, K), 'FontSize', 16, 'FontWeight', 'bold');
fprintf('  Main result created (Figure 3)\n\n');

% Additional Visualization - Color-Coded Segmentation
fprintf('Creating color-coded visualization (for easier interpretation): \n');

figure(4); clf;
set(gcf, 'Position', [100, 100, 1200, 600]);

subplot(1,2,1);
imshow(uint8(imdata));
title('Original Image', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1,2,2);
best_label_rgb = label2rgb(best_labels, 'jet', 'k');
imshow(best_label_rgb);
title(sprintf('Color-Coded Segmentation (M=%d)', best_M), 'FontSize', 14, 'FontWeight', 'bold');
sgtitle('Color-Coded Version (Additional Visualization for Clarity)', 'FontSize', 16, 'FontWeight', 'bold');

% Individual Segments
fprintf('Creating individual segment views: \n');

figure(5); clf;
n_cols = ceil(sqrt(best_M));
n_rows = ceil(best_M / n_cols);

for m = 1:best_M
    subplot(n_rows, n_cols, m);
    
    % Create mask
    mask = (best_labels == m);
    
    % Extract segment
    seg_img = imdata;
    for ch = 1:3
        channel = seg_img(:,:,ch);
        channel(~mask) = 0;
        seg_img(:,:,ch) = channel;
    end
    
    imshow(uint8(seg_img));
    n_pixels = sum(mask(:));
    title(sprintf('Component %d\n%.1f%% (%d pixels)', m, 100*n_pixels/N, n_pixels), 'FontSize', 10);
end
sgtitle(sprintf('Individual Segments (M=%d)', best_M), 'FontSize', 14, 'FontWeight', 'bold');