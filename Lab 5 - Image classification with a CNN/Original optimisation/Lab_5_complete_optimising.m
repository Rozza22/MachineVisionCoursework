% Script for lab 5 - CImage classification with a CNN - optimised from the
% LHS method
clear
close all;
rng(123)

%% Task 1 Designing and training a CNN

% Load the Digits dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc), [32, 32]);
% Split the data into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

numFilters = 10;
filterSize = 6;
numNeurons = 100;
initialLearnRate = 0.000253;


% Define the LeNet-5 architecture
layers = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
         convolution2dLayer(filterSize,numFilters,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
         reluLayer() % ReLU activation function may not be optimal, but put it after convolutional layers if anywhere
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2
         convolution2dLayer(filterSize,2*numFilters,'Padding','same','Name','conv_2')
         reluLayer()
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')
         fullyConnectedLayer(numNeurons,'Name','fc_1') % connects every neuron in previous layer to current one
         fullyConnectedLayer(numNeurons/2,'Name','fc_2')
         fullyConnectedLayer(10,'Name','fc_3')
         softmaxLayer('Name','softmax') % Softmax activation for classification
         classificationLayer('Name','output')];

% Specify the training options
options = trainingOptions('sgdm', ...
 'InitialLearnRate',initialLearnRate, ...
 'MaxEpochs',10, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');

% Train the network
net = trainNetwork(imdsTrain,layers,options);

%% Model accuracy

% Classify validation images and compute accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy of the network on the validation images: %f\n', accuracy);

% Working out the other metrics of accuracy
% Calculate confusion matrix
C = confusionmat(YValidation, YPred);

% Calculate true positive (TP), false positive (FP), and false negative (FN) counts
TP = diag(C); % True positives (correctly predicted)
FP = sum(C, 1)' - TP; % False positives
FN = sum(C, 2) - TP; % False negatives

% Calculate precision and recall
precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);

% Display precision and recall for each class
for i = 1:numel(unique(YValidation))
    fprintf('Class %d - Precision: %.2f, Recall: %.2f\n', i, precision(i), recall(i));
end

% Calculate overall precision and recall (macro-average)
precision = mean(precision);
recall = mean(recall);

% F1 score
F1score = 2*(recall*precision)/(recall+precision);
     