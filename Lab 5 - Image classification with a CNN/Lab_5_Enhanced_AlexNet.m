% lab 5 - CImage classification with a CNN - Architecture exploration
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

%% Task 1 

% Define the AlexNet architecture for the digits dataset
layers = [
    imageInputLayer([32 32 1])              % Input layer for 32x32 grayscale images
    convolution2dLayer(11,96,'Stride',4)     % 1st Convolutional Layer
    reluLayer()                              % ReLU activation function
    maxPooling2dLayer(2,'Stride',2)          % Max pooling layer
    convolution2dLayer(5,256,'Stride',1,'Padding',2) % 2nd Convolutional Layer
    reluLayer()                              % ReLU activation function
    maxPooling2dLayer(2,'Stride',2)          % Max pooling layer
    convolution2dLayer(3,384,'Stride',1,'Padding',1) % 3rd Convolutional Layer
    reluLayer()                              % ReLU activation function
    convolution2dLayer(3,384,'Stride',1,'Padding',1) % 4th Convolutional Layer
    reluLayer()                              % ReLU activation function
    convolution2dLayer(3,256,'Stride',1,'Padding',1) % 5th Convolutional Layer
    reluLayer()                              % ReLU activation function
%     maxPooling2dLayer(2,'Stride',2)          % Max pooling layer
    fullyConnectedLayer(4096)                % 1st Fully connected layer
    reluLayer()                              % ReLU activation function
    dropoutLayer(0.5)                        % Dropout layer
    fullyConnectedLayer(4096)                % 2nd Fully connected layer
    reluLayer()                              % ReLU activation function
    dropoutLayer(0.5)                        % Dropout layer
    fullyConnectedLayer(10)                  % Output layer with 10 classes (digits 0-9)
    softmaxLayer()                           % Softmax activation
    classificationLayer()                    % Classification layer
];

% Specify the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
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
