% LHS optimisation

% L2 regularisation optimisation code
clear
close all;

rng(123); % Set the random number generator seed to 0 (or any other desired seed)

%% Getting data in

% Load the Digits dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc), [32, 32]);
% Split the data into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Finding optimal hyperparameters for the model

% Define parameter ranges (adjust these as needed)
lambdaRange = [0.0001, 0.01]; % Range for number of filters

% Define the number of samples
numSamples = 10; % Adjust as needed

% Perform Latin Hypercube Sampling
parameterSamples = lhsdesign(numSamples, 3); % 3 parameters to sample

for i =1:numSamples

    lambda1 = parameterSamples(i, 1);
    lambda2 = parameterSamples(i, 2);
    lambda3 = parameterSamples(i, 3);

    % Define the LeNet-5 architecture
    layers1 = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
             convolution2dLayer(5,6,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
             averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2
    
             convolution2dLayer(5,16,'Padding','same','Name','conv_2')
             averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')

             fullyConnectedLayer(120,'Name','fc_1','WeightL2Factor', lambda1)
             fullyConnectedLayer(84,'Name','fc_2', 'WeightL2Factor', lambda2)
             fullyConnectedLayer(10,'Name','fc_3', 'WeightL2Factor', lambda3)
    
             softmaxLayer('Name','softmax') % Softmax activation for classification
             classificationLayer('Name','output')];
    
    % Specify the training options
    options1 = trainingOptions('sgdm', ...
     'InitialLearnRate',0.0001, ...
     'MaxEpochs',10, ...
     'Shuffle','every-epoch', ...
     'ValidationData',imdsValidation, ...
     'ValidationFrequency',30, ...
     'Verbose',false, ...
     'Plots','training-progress');
    
    % Train the network
    net1 = trainNetwork(imdsTrain,layers1,options1);

    % Classify validation images and compute accuracy
    YPred = classify(net1, imdsValidation);
    YValidation = imdsValidation.Labels;
    accuracy(i) = sum(YPred == YValidation) / numel(YValidation);

end

% Identify the best set of parameters
bestAccuracy = max(accuracy);
bestParametersIndex = find(accuracy == bestAccuracy);
bestParameters = parameterSamples(bestParametersIndex, :);
fprintf('Best Accuracy: %f\n', bestAccuracy);
fprintf('Best Parameters: %f, %f, %f, %f\n', bestParameters);

%% Model accuracy

% Classify validation images and compute accuracy
YPred = classify(net1,imdsValidation);
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