% Updated Lab1part2

clear
close all;

% reading in chosen image
fruits = imread('Images\Fruits.jpg');
imwrite(fruits,'Figures\imageFruits.png')

% converting image to grayscale in prep for the rest
fruits = rgb2gray(fruits);
fruitsHist = imhist(fruits);

% Applying histogram equalisation
fruitsHistEq = histeq(fruits);

figure
subplot(2,2,1), imshow(fruits), title('Original image in greyscale')
subplot(2,2,3), imshow(fruitsHistEq), title('Image enhanced with histogram equalisation')
subplot(2,2,2), imhist(fruits), title('Original image histogram')
subplot(2,2,4), imhist(fruitsHistEq), title('Histogram of enhanced image')

%% Gamma correction

gamma = 0.7;

fruitsGamma = imadjust(fruits, [], [], gamma);

figure
subplot(2,2,1), imshow(fruits), title('original image')
subplot(2,2,3), imshow(fruitsGamma), title('gamma corrected image')
subplot(2,2,2), imhist(fruits), title('Original image histogram')
subplot(2,2,4), imhist(fruitsGamma), title('Histogram of enhanced image')

%% Adding noise 
% Add Gaussian noise
fruitGaussianNoise = imnoise(fruits, 'gaussian', 0, 0.01);

% Add salt and pepper noise
fruitSandPnoise = imnoise(fruits, 'salt & pepper', 0.05);

%% taking noise away with Gaussian filter

GaussMinusGauss = imgaussfilt(fruitGaussianNoise, 5); % number represents s.d
SandPminusGauss = imgaussfilt(fruitSandPnoise, 5); % number represents s.d

figure 
subplot(2,2,1), imshow(fruitGaussianNoise), title('Image synthesised with Gaussian noise'), zoom(4)
subplot(2,2,2), imshow(fruitSandPnoise), title('Image synthesised with Salt and Pepper noise'), zoom(4)
subplot(2,2,3), imshow(GaussMinusGauss), title('Gaussian filter applied to gaussian noise'), zoom(4)
subplot(2,2,4), imshow(SandPminusGauss), title('Gaussian filter applied to Salt and Pepper noise'), zoom(4)

%% Taking away Salt and Pepper noise with median filter

SandPminusMedian = medfilt2(fruitSandPnoise, [1 1]);
figure
subplot(1,2,1), imshow(fruitSandPnoise), title('Image with Salt and Pepper noise'), zoom(4)
subplot(1,2,2), imshow(SandPminusMedian), title('Image filtered using median filter'), zoom(4)

%% Sobel edge detection
BWsobelInit = edge(fruits, "sobel");
BWsobelsecond = edge(fruits, "sobel", 0.03);
BWsobelOptimal = edge(fruits, "sobel", 0.03, "nothinning"); % This uses the sobel edge detection method

figure
subplot(3,3,1), imshow(BWsobelInit), title('Default Sobel edge detection')
subplot(3,3,2), imshow(BWsobelsecond), title('Optimised Sobel edge detection')
subplot(3,3,3), imshow(BWsobelOptimal), title('Optimised Sobel edge detection with nothinning on')
%% Prewitt
BWprewittInit = edge(fruits, "prewitt");
BWprewittsecond = edge(fruits, "prewitt", 0.03);
BWprewittOptimal = edge(fruits, "prewitt", 0.03, "nothinning");

% figure
subplot(3,3,4), imshow(BWprewittInit), title('Default Prewitt edge detection')
subplot(3,3,5), imshow(BWprewittsecond), title('Optimised Prewitt edge detection')
subplot(3,3,6), imshow(BWprewittOptimal), title('Optimised Prewitt edge detection with nothinning on')
%% Canny 
sigma = 0.1;
BWCannyInit = edge(fruitsHistEq, 'Canny');
BWCannySecond = edge(fruitsHistEq, 'Canny', sigma);
BWCannyOptimal = edge(fruitsHistEq, 'Canny',[0.03 0.07], sigma);

% figure
subplot(3,3,7), imshow(BWCannyInit), title('Default Canny edge detection')
subplot(3,3,8), imshow(BWCannySecond), title('Optimised Canny edge detection')
subplot(3,3,9), imshow(BWCannyOptimal), title('Optimised Canny edge detection with nothinning on')