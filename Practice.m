% Lab 1 Script
% Make sure you convert all variables to uint8 for image stuff

im = imread('Lab 1 - Part I - Introduction to Images and Videos/Images/Apples.jpg');

im(2,3) % Tells us the intensity at pixel of row 2 column 3

% RGB(2,3,:) %Supposed to return RGB value but doesn't do anything
% red_pixel = nnz(im(:, :, 1) == 237 & ... % im(:, :, 2) == 1   & ...
%                 im(:, :, 3) == 12); % Supposed to return RGB

figure
subplot(1,2,1)
imshow(im) % displays RGB image
title('image')
[r, c] = size(im); % Size of image
info = imfinfo('Lab 1 - Part I - Introduction to Images and Videos/Images/Apples.jpg'); % Returns Information about the image

%% Extract RGB colours from within
% Extract blue frame of data and display
% figure
imageB = im(:,:,3);
subplot(1,2,2)
imshow(imageB)
title('Image Data - Blue')

% % Extract blue frame of data and display
% imageG = im(:,:,2);
% subplot(1,3,2)
% imshow(imageG)
% title('Image Data - Green')
% 
% % Extract blue frame of data and display
% imageR = im(:,:,1);
% subplot(1,3,1)
% imshow(imageR)
% title('Image Data - Red')

%% Calling function to generate maximum RGB values
[maxR, maxG, maxB] = maxColor(im);

%% Concatenate two images
im1 = imread('Lab 1 - Part I - Introduction to Images and Videos\Images\Beautiful_Green_Picture.jpg');
im2 = imread('Lab 1 - Part I - Introduction to Images and Videos\Images\UAV_Landing.png');

TwoImages = [im1(1:522,:,:);im2(:,1:960,:)]; % Had to adjust dimensions on both images to be the same
imshow(TwoImages)

%% Convert colour image to greyscale

GIm = rgb2gray(im);
[nr,nc,np] = size(GIm);
imshow(GIm)

FlippedImage = flipLtRt(im);
imshow(FlippedImage)
GreyFlippedIm = rgb2gray(FlippedImage);

%% Gamma Correction
% [X, map] = imread("Lab 1 - Part I - Introduction to Images and
% Videos\Dog.jpg"); % Does this need to be .tif?
% I = ind2gray(X, map);
% J = imadjust(I,[],[],0.5);
% imshow(I)
% figure
% imshow(J)

%% Histogram of image 
HistIm1 = imhist(im1);
imhist(im1)

%% Writing images in MATLAB
imwrite(GreyFlippedIm, "GrFlApples.jpg")
imfinfo GrFlApples.jpg % information on image file

%% Changing brightness of images
figure
subplot(1,2,1)
imshow(GreyFlippedIm)

L_Br = GreyFlippedIm-100; % decreases image brightness by 100 intensity
subplot(1,2,2)
imshow(L_Br)

%% changing colour of image - using function
clear 

UAV = imread('Lab 1 - Part I - Introduction to Images and Videos/Images/UAV.jpg');
YellowUAV = changeColour(UAV);

figure
subplot(1,2,1)
imshow(UAV)
subplot(1,2,2)
imshow(YellowUAV)

%% Finding an area of predefined colour
twoColour = imread('Lab 1 - Part I - Introduction to Images and Videos/Two_colour.jpg');

% Extract RGB channels separately
red_channel = twoColour(:,:,1);
green_channel = twoColour(:,:,2);
blue_channel = twoColour(:,:,3);

% label pixels of yellow colour
yellow_map = green_channel>150 & red_channel>150 & blue_channel<50; % Can see different colours in RGB  form by just looking it up
[iYellow, jYellow] = find(yellow_map > 0);

figure
subplot(1,2,1)
imshow(twoColour)

subplot(1,2,2)
imshow(twoColour)
hold on;
scatter(jYellow,iYellow,5,'filled') % function uses spatial coordinates are reverse so we have to swap i and j round 
title('')

%% Conversion between different Formats
clear
robot = imread('Lab 1 - Part I - Introduction to Images and Videos/Images/Royalty Free Robot Picture.jpg');
robotGrey = rgb2gray(robot);
robotHSV = rgb2hsv(robot);
robotBinary = imbinarize(robotGrey);
robotLight = rgb2lightness(robot);
[robotIndexed, mapRobot] = rgb2ind(robot,16);

figure
subplot(2,3,1)
imshow(robot)
title('Original')

subplot(2,3,2)
imshow(robotGrey)
title('Greyscale')

subplot(2,3,3)
imshow(robotHSV)
title('HueSaturationValue')

subplot(2,3,4)
imshow(robotBinary)
title('Binary')

subplot(2,3,5)
imshow(robotLight,[])
title('Lightness of Image')

subplot(2,3,6)
imagesc(robotIndexed)
title('Indexed')
colormap(mapRobot)
axis image
% zoom(4)

%% Understanding Image Histogram testing
clear
landscape = imread('Lab 1 - Part I - Introduction to Images and Videos/Images/Landscape.jpg');
landscapeGrey = rgb2gray(landscape);

figure
subplot(4,1,1)
imhist(landscapeGrey)
title("Using 'imhist'")
xlabel('Number of bins (256 by default for a greyscale image)')
ylabel('Histogram counts')

h = imhist(landscapeGrey);
h1 = h(1:10:256);
horz = 1:10:256;
subplot(4,1,2)
bar(horz,h1)

subplot(4,1,3)
stem(landscapeGrey)

subplot(4,1,4)
plot(h)

% have part 2 of this section done but this is a different way. I think the
% way I did it is better
r = double(landscape(:,:,1));
g = double(landscape(:,:,2));
b = double(landscape(:,:,3));
figure
subplot(3,1,1)
hist(r(:),124)
title('Histogram of the red colour')
subplot(3,1,2)
hist(g(:),124)
title('Histogram of the green colour')
subplot(3,1,3)
hist(b(:),124)
title('Histogram of the blue colour')
