% Lab 2 
clear
close all;

Ginger1 = imread('GingerBreadMan_first.jpg'); % Read in first and second gingerbread man images
Ginger2 = imread('GingerBreadMan_second.jpg');

redSquare = imread('red_square_static.jpg'); % Read in red square image

Ginger1grey = rgb2gray(Ginger1); % Convert all to grayscale to allow operations on them
Ginger2grey = rgb2gray(Ginger2);
redSquareGrey = rgb2gray(redSquare);

Ginger1Corner = corner(Ginger1grey, 1); % Find corners on all images
% Ginger2Corner = corner(Ginger2grey(Ginger1Corner:Ginger1Corner+SizeGinger), 1);
redSquareCorner = corner(redSquareGrey);
redSquareCorner = redSquareCorner(1,:); % Pick out the corner I want


SizeGinger = 50; % Size of bounding boxes I wanted for my figure to make it clearer
SizeSquare = 20;

bboxGinger1 = [Ginger1Corner(1,1)-SizeGinger/2, Ginger1Corner(1,2)-SizeGinger/2, SizeGinger, SizeGinger];
bboxSquare = [redSquareCorner-SizeSquare/2, SizeSquare, SizeSquare]; % Creating bounding boxes

markedGinger1 = insertShape(Ginger1grey, "filled-rectangle", bboxGinger1, 'Color', 'red');
markedGinger1 = insertMarker(markedGinger1, Ginger1Corner); % adds marks to the images so it can be seen in figures
markedSquare = insertShape(redSquareGrey, "filled-rectangle", bboxSquare, "Color", "red");
markedSquare = insertMarker(markedSquare, redSquareCorner);

markedCorners = figure; % Necessary step to save image where I want it
subplot(1,2,1), imshow(markedGinger1), zoom(2)
subplot(1,2,2), imshow(markedSquare), zoom(4)
saveas(markedCorners,'markedCorners.png')

Ginger2Corner = corner(Ginger2grey(Ginger1Corner(1,1):Ginger1Corner(1,1)+SizeGinger, ...
    Ginger1Corner(1,2):Ginger1Corner(1,2)+SizeGinger), 1);

%% Trying to visualise optical flow
opticFlow = opticalFlowLK("NoiseThreshold",0.005); %  initializes an object that can be used to compute optical flow using the LK algorithm
flow1 = estimateFlow(opticFlow, Ginger1grey); % computes the optical flow
flow2 = estimateFlow(opticFlow, Ginger2grey);

% Second gingerbread man
h2 = figure;
movegui(h2);
hViewPanel = uipanel(h2,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

imshow(Ginger2grey), title('flow of Gingerbread man first')
hold on
plot(flow2,'DecimationFactor',[10 10],'ScaleFactor',10,'Parent',hPlot)
hold off
saveas(h2,'OpticalFlow.png')

%% Corner of sqaure
clear
figure
redSquare = imread('red_square_static.jpg');
redSquareGrey = rgb2gray(redSquare);
redSquareCorner = corner(redSquareGrey);

MinA = min(redSquareCorner(:,1,1));
MinB = min(redSquareCorner(:,2,1));
TopLeftCorner = [MinA,MinB];

SizeSquare = 20;
bboxSquare = [TopLeftCorner(1,1)-SizeSquare/2, TopLeftCorner(1,2)-SizeSquare/2, SizeSquare, SizeSquare];

markedSquare = insertShape(redSquareGrey, "filled-rectangle", bboxSquare, "Color", "green");
imshow(markedSquare), title('Box shows region of corner')

%% Optical flow for video format
clear

videoReader = VideoReader('red_square_video.mp4'); % Creates video reader object
numFrames = 0;
while hasFrame(videoReader) % Works out how many frames there are so I know how many times to run for loop later
    readFrame(videoReader);
    numFrames = numFrames + 1;
end

opticFlow = opticalFlowLK("NoiseThreshold",0.005); %  initializes an object that can be used to compute optical flow using the LK algorithm

videoReader = VideoReader('red_square_video.mp4'); % overwrites old videoReader object

% % Need to put this in loop to get the last one

% First frame
video = readFrame(videoReader);
videoFrame = rgb2gray(video);
SquareCorner = corner(videoFrame);
MinA = min(SquareCorner(:,1,1));
MinB = min(SquareCorner(:,2,1));
TopLeftCorner = [MinA,MinB]; % Works out top left corner, this is the point to be tracked

position = zeros(numFrames,2); % initialising array of positions which will be filled later
position(1,:) = TopLeftCorner;

for i = 1:numFrames-1 % goes through video frame by frame then marks the coordinates of the top left corner
    video = readFrame(videoReader);
    videoFrame = rgb2gray(video);
    SquareCorner = corner(videoFrame);
    NearestCorner = dsearchn(SquareCorner,position(i,:));
    NearestCorner = SquareCorner(NearestCorner,:);
    flow = estimateFlow(opticFlow,videoFrame);
    corner_x = NearestCorner(1,1);
    corner_y = NearestCorner(1,2);
    x_new = corner_x + flow.Vx(round(corner_y), round(corner_x));
    y_new = corner_y + flow.Vy(round(corner_y), round(corner_x));
    position(i+1,:) = [x_new,y_new]; % Filling out array with positions
end

%% Compare trajectory to real data

load("red_square_gt.mat")

% plot created data
trackCorner = figure;
scatter(position(:,1),position(:,2),".","red")
hold on
scatter(gt_track_spatial(:,1),gt_track_spatial(:,2),".","green")
hold off
legend('Tracked Trajectory','Ground Truth')
saveas(trackCorner,'trackCorner.png')

RMSE = rmse(gt_track_spatial,position); % Calculating RMSE between my results and the actual position
RMSEt = sqrt(RMSE(1,1)^2 + RMSE(1,2)^2);

% Data in folder is not accurate to video but each data point is accurate
% relative to the last one
