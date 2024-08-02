% MATLAB example for single point tracking
clear

% Create system objects for reading and displaying video and for drawing a
% bounding box of the object
videoReader = VideoReader('visionface.avi');
videoPlayer = vision.VideoPlayer('Position',[100,100,680,520]);

% Read the first video frame
objectFrame = readFrame(videoReader);
objectRegion = [264,122,93,93]; % position of face is known

% the following allows you to select region with mouse
% figure; imshow(objectFrame);
% objectRegion=round(getPosition(imrect))

% Show initial frame with a red bounding box
objectImage = insertShape(objectFrame,'rectangle',objectRegion,'Color','red');
figure;
imshow(objectImage);
title('Red box shows object region');

% Detect interest points in the region
points = detectMinEigenFeatures(im2gray(objectFrame),'ROI',objectRegion);

% Display the detected points
pointImage = insertMarker(objectFrame,points.Location,'+','Color','white');
figure;
imshow(pointImage);
title('Detected interest points');

% Create a tracker object
tracker = vision.PointTracker('MaxBidirectionalError',1);

% initialize the tracker
initialize(tracker,points.Location,objectFrame);

% Read, track, display points, and results in each video frame.
while hasFrame(videoReader)
      frame = readFrame(videoReader);
      [points,validity] = tracker(frame);
      out = insertMarker(frame,points(validity, :),'+');
      videoPlayer(out);
end

% Release the video player
release(videoPlayer);

%% My own try - not the way it says in the guidance
clear

redSquare = imread('red_square_static.jpg');
redSquareGrey = rgb2gray(redSquare);
redSquareCorner = corner(redSquareGrey);

% put for loop here to get top left corner of square8
% size(redSquareCorner,1)
% for i = 1:size(redSquareCorner,1)
MinA = min(redSquareCorner(:,1,1));
MinB = min(redSquareCorner(:,2,1));
TopLeftCorner = [MinA,MinB];


videoReader = VideoReader('red_square_video.mp4');
videoPlayer = vision.VideoPlayer('Position',[100,100,680,520]);
objectFrame = readFrame(videoReader);
% objectRegion = [264,122,93,93]; % is this the same the bbox?

SizeSquare = 20;
bboxSquare = [TopLeftCorner(1,1)-SizeSquare/2, TopLeftCorner(1,2)-SizeSquare/2, SizeSquare, SizeSquare];

markedSquare = insertShape(redSquareGrey, "filled-rectangle", bboxSquare, "Color", "green");
imshow(markedSquare), title('Box shows region of corner')

% pointTracker = vision.PointTracker('MaxBdirectionalError',1);
pointTracker = vision.PointTracker;
% [TopLeftCorner,L] = PointTracker(redSquareGrey);

% initialize(pointTracker, TopLeftCorner.Location,redSquareGrey)
initialize(pointTracker, TopLeftCorner, objectFrame)

while hasFrame(videoReader)
      frame = readFrame(videoReader);
      % frame = rgb2gray(frame);
      [points,validity] = pointTracker(frame);
      out = insertMarker(frame,points(validity, :),'+');
      videoPlayer(out);
end

release(videoPlayer)

%% Method for tracking all points
clear

redSquare = imread('red_square_static.jpg');
redSquareGrey = rgb2gray(redSquare);
redSquareCorner = corner(redSquareGrey);

vidReader = VideoReader('red_square_video.mp4','CurrentTime',1);
opticFlow = opticalFlowLK;
h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGray = im2gray(frameRGB);  
    flow = estimateFlow(opticFlow,frameGray);
    imshow(frameRGB)
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',60,'Parent',hPlot);
    hold off
    pause(10^-3)
end

% Data in folder is not accurate to video but each data point is accurate
% relative to the last one