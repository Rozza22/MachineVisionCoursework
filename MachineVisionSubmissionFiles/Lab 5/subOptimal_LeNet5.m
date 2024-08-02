% Sub-optimal LeNet-5 CNN - With all accuracy scores calculated at the end
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

%% Sub-optimal LeNet-5 CNN

% Define the LeNet-5 architecture
layers3 = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
         convolution2dLayer(3,64,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2

         convolution2dLayer(5,32,'Padding','same','Name','conv_2')
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')

         fullyConnectedLayer(256,'Name','fc_1') % connects every neuron in previous layer to current one
         fullyConnectedLayer(256,'Name','fc_2')
         fullyConnectedLayer(10,'Name','fc_3')
         softmaxLayer('Name','softmax') % Softmax activation for classification
         classificationLayer('Name','output')];

% Specify the training options
options3 = trainingOptions('sgdm', ...
 'InitialLearnRate',0.0001, ...
 'MaxEpochs',20, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');

% Train the network
net3 = trainNetwork(imdsTrain,layers3,options3);

%% Model accuracy

% Classify validation images and compute accuracy
YPred = classify(net3,imdsValidation);
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