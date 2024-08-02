% All models in here - will take about an hour to run so I suggest running
% the other files in this folder

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

%% AlexNet code with hyperparameters

% Define the architecture
layers2 = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
         convolution2dLayer(3,96,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
         maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2

         convolution2dLayer(5,256,'Padding','same','Name','conv_2')
         maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')

         convolution2dLayer(3,384,'Padding','same','Name','conv_3')
         convolution2dLayer(3,384,'Padding','same','Name','conv_4')
         convolution2dLayer(3,256,'Padding','same','Name','conv_5')
         maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')

         fullyConnectedLayer(4096,'Name','fc_1') % connects every neuron in previous layer to current one
         fullyConnectedLayer(4096,'Name','fc_2')

         fullyConnectedLayer(10)

         softmaxLayer('Name','softmax') % Softmax activation for classification
         classificationLayer('Name','output')];

% Specify the training options
options2 = trainingOptions('sgdm', ...
 'InitialLearnRate',0.0001, ...
 'MaxEpochs',10, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');

% Train the network
net2 = trainNetwork(imdsTrain,layers2,options2);

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

%% Optimal LeNet-5 CNN

% Define the LeNet-5 architecture
layers4 = [imageInputLayer([32 32 1],'Name','input') % Input layer for 32x32 grayscale images
         convolution2dLayer(5,6,'Padding','same','Name','conv_1') % second number is number of filters, first is the size of these filters
         reluLayer()
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_1') % reduces the spatial dimensions of the feature maps, 2x2 average pooling with stride 2
         convolution2dLayer(5,16,'Padding','same','Name','conv_2')
         reluLayer()
         averagePooling2dLayer(2,'Stride',2,'Name','avgpool_2')
         fullyConnectedLayer(120,'Name','fc_1') % connects every neuron in previous layer to current one
         fullyConnectedLayer(84,'Name','fc_2')
         fullyConnectedLayer(10,'Name','fc_3')
         softmaxLayer('Name','softmax') % Softmax activation for classification
         classificationLayer('Name','output')];

% Specify the training options
options4 = trainingOptions('sgdm', ...
 'InitialLearnRate',0.0001, ...
 'MaxEpochs',10, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');

% Train the network
net4 = trainNetwork(imdsTrain,layers4,options4);