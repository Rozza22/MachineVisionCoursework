% Attempt at Latin Hypercube sampling to optimise KNN
%% initialise
clear;
close all;
rng(123)

% Load the Digits dataset
digitDatasetPath = fullfile(toolboxdir('nnet'), 'nndemos', ...
 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc), [32, 32]);
% Split the data into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% make models and choose

% Define parameter ranges (adjust these as needed)
numFiltersRange = [4, 32]; % Range for number of filters
filterSizeRange = [3, 7];  % Range for filter size
numNeuronsRange = [50, 200]; % Range for number of neurons in fully connected layers
initialLearnRateRange = [0.0001, 0.001]; % Range for initial learning rate
StrideLengthRange = [1, 4];

% Define the number of samples
numSamples = 10; % Adjust as needed

% Perform Latin Hypercube Sampling
parameterSamples = lhsdesign(numSamples, 5); % 5 parameters to sample

% Train the network for each sampled point
for i = 1:numSamples
    % Extract sampled parameters
    numFilters = round(parameterSamples(i, 1) * (numFiltersRange(2) - numFiltersRange(1)) + numFiltersRange(1));
    filterSize = round(parameterSamples(i, 2) * (filterSizeRange(2) - filterSizeRange(1)) + filterSizeRange(1));
    numNeurons = round(parameterSamples(i, 3) * (numNeuronsRange(2) - numNeuronsRange(1)) + numNeuronsRange(1));
    initialLearnRate = parameterSamples(i, 4) * (initialLearnRateRange(2) - initialLearnRateRange(1)) + initialLearnRateRange(1);
    strideLength = round(parameterSamples(i,5) * (StrideLengthRange(2) - StrideLengthRange(1)) + StrideLengthRange(1));
    
    % Define the LeNet-5 architecture with sampled parameters
    layers = [
        imageInputLayer([32 32 1],'Name','input')
        convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv_1')
        averagePooling2dLayer(2, 'Stride', strideLength, 'Name', 'avgpool_1')
        convolution2dLayer(filterSize, 2*numFilters, 'Padding', 'same', 'Name', 'conv_2')
        averagePooling2dLayer(2, 'Stride', strideLength, 'Name', 'avgpool_2')
        fullyConnectedLayer(numNeurons, 'Name', 'fc_1')
        fullyConnectedLayer(round(numNeurons/2), 'Name', 'fc_2')
        fullyConnectedLayer(10, 'Name', 'fc_3')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % Specify the training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', initialLearnRate, ...
        'MaxEpochs', 10, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', imdsValidation, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the network
    net = trainNetwork(imdsTrain, layers, options);
    
    % Classify validation images and compute accuracy
    YPred = classify(net, imdsValidation);
    YValidation = imdsValidation.Labels;
    accuracy(i) = sum(YPred == YValidation) / numel(YValidation);

%                     closeTrainingPlot;
end

% Identify the best set of parameters
bestAccuracy = max(accuracy);
bestParametersIndex = find(accuracy == bestAccuracy);
bestParameters = parameterSamples(bestParametersIndex, :);
fprintf('Best Accuracy: %f\n', bestAccuracy);
fprintf('Best Parameters: %f, %f, %f, %f, %f\n', bestParameters);

numFilters = round(bestParameters(1,1) * (numFiltersRange(2) - numFiltersRange(1)) + numFiltersRange(1));
filterSize = round(bestParameters(1,2) * (filterSizeRange(2) - filterSizeRange(1)) + filterSizeRange(1));
numNeurons = round(bestParameters(1,3) * (numNeuronsRange(2) - numNeuronsRange(1)) + numNeuronsRange(1));
initialLearnRate = bestParameters(1,4) * (initialLearnRateRange(2) - initialLearnRateRange(1)) + initialLearnRateRange(1);
strideLength = round(bestParameters(1,5) * (StrideLengthRange(2) - StrideLengthRange(1)) + StrideLengthRange(1));