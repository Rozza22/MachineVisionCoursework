% Hyperparamter tuning
clear;
close all;
rng(123)

%% Load the Digits dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc), [32, 32]);
% Split the data into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Task 1 Designing and training a CNN

% Define the LeNet-5 architecture with adjusted hyperparameters and regularization
numFilters = 64; % number of filters
filterSize = 5; % filter size 
numNeurons = 512; % number of neurons
initialLearnRate = 0.0001; % initial learning rate
strideLength = 2; % stride length

% Define the LeNet-5 architecture with sampled parameters
layers1 = [
    imageInputLayer([32 32 1],'Name','input')
    convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv_1')
%     maxPooling2dLayer(2, 'Stride', strideLength, 'Name', 'maxpool_1') % Use max pooling for better feature selection
    averagePooling2dLayer(2, 'Stride', strideLength, 'Name', 'maxpool_1')
    convolution2dLayer(filterSize, 2*numFilters, 'Padding', 'same', 'Name', 'conv_2')
%     maxPooling2dLayer(2, 'Stride', strideLength, 'Name', 'maxpool_2') % Use max pooling for better feature selection
    averagePooling2dLayer(2, 'Stride', strideLength, 'Name', 'maxpool_2') % Use max pooling for better feature selection

    fullyConnectedLayer(numNeurons, 'Name', 'fc_1')
    dropoutLayer(0.2, 'Name', 'dropout_1') % Add dropout for regularization
    fullyConnectedLayer(numNeurons, 'Name', 'fc_2')
    dropoutLayer(0.2, 'Name', 'dropout_2') % Add dropout for regularization
    fullyConnectedLayer(10, 'Name', 'fc_3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify the training options with adjusted regularization
options1 = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', 20, ... % Increase the number of epochs for better convergence
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net1 = trainNetwork(imdsTrain, layers1, options1);


%% Model accuracy

% Classify validation images and compute accuracy
YPred1 = classify(net1,imdsValidation);
YValidation1 = imdsValidation.Labels;
accuracy1 = sum(YPred1 == YValidation1)/numel(YValidation1);
fprintf('Accuracy of the network on the validation images: %f\n', accuracy1);

% Working out the other metrics of accuracy
% Calculate confusion matrix
C1 = confusionmat(YValidation1, YPred1);

% Calculate true positive (TP), false positive (FP), and false negative (FN) counts
TP1 = diag(C1); % True positives (correctly predicted)
FP1 = sum(C1, 1)' - TP1; % False positives
FN1 = sum(C1, 2) - TP1; % False negatives

% Calculate precision and recall
precision1 = TP1 ./ (TP1 + FP1);
recall1 = TP1 ./ (TP1 + FN1);

% Display precision and recall for each class
for i = 1:numel(unique(YValidation1))
    fprintf('Class %d - Precision: %.2f, Recall: %.2f\n', i, precision1(i), recall1(i));
end

% Calculate overall precision and recall (macro-average)
precision1 = mean(precision1);
recall1 = mean(recall1);

% F1 score
F1score1 = 2*(recall1*precision1)/(recall1+precision1);

%% Task 1 Designing and training a CNN - second one

% % Define the LeNet-5 architecture
% layers2 = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
%          convolution2dLayer(3,64,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
%          averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2
% 
%          convolution2dLayer(5,32,'Padding','same','Name','conv_2')
%          averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')
% 
%          fullyConnectedLayer(256,'Name','fc_1') % connects every neuron in previous layer to current one
%          fullyConnectedLayer(256,'Name','fc_2')
%          fullyConnectedLayer(10,'Name','fc_3')
%          softmaxLayer('Name','softmax') % Softmax activation for classification
%          classificationLayer('Name','output')];
% 
% % Specify the training options
% options2 = trainingOptions('sgdm', ...
%  'InitialLearnRate',0.0001, ...
%  'MaxEpochs',20, ...
%  'Shuffle','every-epoch', ...
%  'ValidationData',imdsValidation, ...
%  'ValidationFrequency',30, ...
%  'Verbose',false, ...
%  'Plots','training-progress');
% 
% % Train the network
% net2 = trainNetwork(imdsTrain,layers2,options2);
% 
% %% Model accuracy
% 
% % Classify validation images and compute accuracy
% YPred2 = classify(net2,imdsValidation);
% YValidation2 = imdsValidation.Labels;
% accuracy2 = sum(YPred2 == YValidation2)/numel(YValidation2);
% fprintf('Accuracy of the network on the validation images: %f\n', accuracy2);
% 
% % Working out the other metrics of accuracy
% % Calculate confusion matrix
% C2 = confusionmat(YValidation2, YPred2);
% 
% % Calculate true positive (TP), false positive (FP), and false negative (FN) counts
% TP2 = diag(C2); % True positives (correctly predicted)
% FP2 = sum(C2, 1)' - TP2; % False positives
% FN2 = sum(C2, 2) - TP2; % False negatives
% 
% % Calculate precision and recall
% precision2 = TP2 ./ (TP2 + FP2);
% recall2 = TP2 ./ (TP2 + FN2);
% 
% % Display precision and recall for each class
% for i = 1:numel(unique(YValidation2))
%     fprintf('Class %d - Precision: %.2f, Recall: %.2f\n', i, precision2(i), recall2(i));
% end
% 
% % Calculate overall precision and recall (macro-average)
% precision2 = mean(precision2);
% recall2 = mean(recall2);
% 
% % F1 score
% F1score2 = 2*(recall2*precision2)/(recall2+precision2);
