% Lab 3 part 2
%% Gaussian part

clear all
close all 

videoReader = VideoReader('car-tracking.mp4');
numFrames = 0;
while hasFrame(videoReader)
    readFrame(videoReader);
    numFrames = numFrames + 1;
end

%% Low n_frames to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 1;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
n_gaussians = 5;   

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    lowNF.fr = readFrame(source);     % read in frame
    
    lowNF.fgMask = step(detector, lowNF.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    lowNF.fg = uint8(zeros(size(lowNF.fr, 1), size(lowNF.fr, 2)));
    lowNF.fg(lowNF.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(lowNF.fr)
    subplot(2,1,2), imshow(lowNF.fg)
    drawnow
    
    writeVideo(output, lowNF.fg);    % save frame into the output video
end


close(output); % save video

%% mid n_frames to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 10;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
n_gaussians = 5;   

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    midNF.fr = readFrame(source);     % read in frame
    
    midNF.fgMask = step(detector, midNF.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    midNF.fg = uint8(zeros(size(midNF.fr, 1), size(midNF.fr, 2)));
    midNF.fg(midNF.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(midNF.fr)
    subplot(2,1,2), imshow(midNF.fg)
    drawnow
    
    writeVideo(output, midNF.fg);    % save frame into the output video
end


close(output); % save video

%% High n_frames to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 30;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
n_gaussians = 5;   

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    highNF.fr = readFrame(source);     % read in frame
    
    highNF.fgMask = step(detector, highNF.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    highNF.fg = uint8(zeros(size(highNF.fr, 1), size(highNF.fr, 2)));
    highNF.fg(highNF.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(highNF.fr)
    subplot(2,1,2), imshow(highNF.fg)
    drawnow
    
    writeVideo(output, highNF.fg);    % save frame into the output video
end


close(output); % save video

%% Compare n_frames 
CompareNFrames = figure;
title('Comparing foreground masks based off changing n_frames learnt on')
subplot(2,2,1), imshow(highNF.fr), title('Frame in black and white')
subplot(2,2,2), imshow(lowNF.fg), title('Low nFrames (1)')
subplot(2,2,3), imshow(midNF.fg), title('Mid nFrames (10)')
subplot(2,2,4), imshow(highNF.fg), title('High nFrames (30)')
saveas(CompareNFrames,'Figures/CompareNFrames.png')

%% Low n_gaussians to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 10;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
% n_gaussians = 1;   % This creates a black screen
n_gaussians = 2; 

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    lowGauss.fr = readFrame(source);     % read in frame
    
    lowGauss.fgMask = step(detector, lowGauss.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    lowGauss.fg = uint8(zeros(size(lowGauss.fr, 1), size(lowGauss.fr, 2)));
    lowGauss.fg(lowGauss.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(lowGauss.fr)
    subplot(2,1,2), imshow(lowGauss.fg)
    drawnow
    
    writeVideo(output, lowGauss.fg);    % save frame into the output video
end


close(output); % save video

%% mid n_gaussians to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 10;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
n_gaussians = 3;   

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    midGauss.fr = readFrame(source);     % read in frame
    
    midGauss.fgMask = step(detector, midGauss.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    midGauss.fg = uint8(zeros(size(midGauss.fr, 1), size(midGauss.fr, 2)));
    midGauss.fg(midGauss.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(midGauss.fr)
    subplot(2,1,2), imshow(midGauss.fg)
    drawnow
    
    writeVideo(output, midGauss.fg);    % save frame into the output video
end


close(output); % save video

%% High n_gaussians to train on

% read the video
source = VideoReader('car-tracking.mp4');  

% create and open the object to write the results
output = VideoWriter('gmm_output.mp4', 'MPEG-4');
open(output);

% create foreground detector object
% a parameter to vary - Number of frames used to compute foreground mask
n_frames = 10;   

% a parameter to vary - Number of Gaussian modes in the mixture model,
% specified as a positive integer
n_gaussians = 5;   

detector = vision.ForegroundDetector('NumTrainingFrames', n_frames, 'NumGaussians', n_gaussians);

% --------------------- process frames -----------------------------------
% loop all the frames
for i = 1:200
    highGauss.fr = readFrame(source);     % read in frame
    
    highGauss.fgMask = step(detector, highGauss.fr);    % compute the foreground mask by Gaussian mixture models
    
    % create frame with foreground detection
    highGauss.fg = uint8(zeros(size(highGauss.fr, 1), size(highGauss.fr, 2)));
    highGauss.fg(highGauss.fgMask) = 255;
    
    % visualise the results
    figure(1),subplot(2,1,1), imshow(highGauss.fr)
    subplot(2,1,2), imshow(highGauss.fg)
    drawnow
    
    writeVideo(output, highGauss.fg);    % save frame into the output video
end


close(output); % save video

%% Compare n_frames 
CompareNGauss = figure;
title('Comparing foreground masks based off changing NumGaussian modes')
subplot(2,2,1), imshow(highGauss.fr), title('Frame')
subplot(2,2,2), imshow(lowGauss.fg), title('Low nGauss (2)')
subplot(2,2,3), imshow(midGauss.fg), title('Mid nGauss (5)')
subplot(2,2,4), imshow(highGauss.fg), title('High Gauss (10)')
saveas(CompareNGauss,'Figures/CompareNGauss.png')