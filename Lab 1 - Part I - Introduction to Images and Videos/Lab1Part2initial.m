% Lab 1 part 2 finalised script

clear
close all;

Fruits = imread('Images\Fruits.jpg');
imwrite(Fruits,'Figures\imageFruits.png')

% Images with noise
ImageGaussian = imnoise(Fruits, 'gaussian'); %adding noise to images
ImageSandP = imnoise(Fruits, 'salt & pepper');

FruitsWithNoise = figure;
subplot(1,2,1), imshow(ImageGaussian), title('Gaussian Synthesisation'), zoom(4)
subplot(1,2,2), imshow(ImageSandP), title('Salt and Pepper Synthesisation'), zoom(4)
saveas(FruitsWithNoise, 'Figures/FruitsWithNoise.png')

Fruits = rgb2gray(Fruits);

% Edge detection methods
BWsobelInit = edge(Fruits, "sobel");
BWsobelsecond = edge(Fruits, "sobel", 0.03);
BWsobelOptimal = edge(Fruits, "sobel", 0.03, "nothinning"); % This uses the sobel edge detection method

% adjusting threshold------------------------------------------------------
BWprewittInit = edge(Fruits, "prewitt");
BWprewittsecond = edge(Fruits, "prewitt", 0.03);
BWprewittOptimal = edge(Fruits, "prewitt", 0.03, "nothinning");

SegmentationThresholdComp = figure;
subplot(3,2,1), imshow(BWsobelInit), title('Sobel operator for segmentation')
subplot(3,2,3), imshow(BWsobelsecond), title('Optimised Sobel segmentation')
subplot(3,2,5), imshow(BWsobelOptimal), title('Optimised Sobel - Nothinning')
subplot(3,2,2), imshow(BWprewittInit), title('Prewitt operator for segmentation')
subplot(3,2,4), imshow(BWprewittsecond), title('Optimised Prewitt segmentation')
subplot(3,2,6), imshow(BWprewittOptimal), title('Optimised Prewitt - Nothinning')
saveas(SegmentationThresholdComp, 'Figures/SegmentationThresholdComp.png')

BWcannyOptimal = edge(Fruits, "canny", [0.025 0.1], "nothinning");

% adjusting noise----------------------------------------------------------
ImageGaussian = imnoise(Fruits, 'gaussian'); 
ImageSandP = imnoise(Fruits, 'salt & pepper');

SobelGaussianEdge1 = edge(ImageGaussian, "sobel", 0.03, "nothinning");
SobelSandPEdge1 = edge(ImageSandP, "sobel", 0.03, "nothinning");
PrewittGaussianEdge1 = edge(ImageGaussian, "prewitt", 0.03, "nothinning");
PrewittSandPEdge1 = edge(ImageSandP, "prewitt", 0.03, "nothinning");

SegmentationNoiseComp = figure;
subplot(3,2,1), imshow(BWsobelOptimal), title('Sobel operator for segmentation')
subplot(3,2,3), imshow(SobelGaussianEdge1), title('Sobel edges on Gaussian noise')
subplot(3,2,5), imshow(SobelSandPEdge1), title('Sobel edges on S and P noise')
subplot(3,2,2), imshow(BWprewittOptimal), title('Prewitt operator for segmentation')
subplot(3,2,4), imshow(PrewittGaussianEdge1), title('Prewitt edges on Gaussian noise')
subplot(3,2,6), imshow(PrewittSandPEdge1), title('Prewitt edges on S and P noise')
saveas(SegmentationNoiseComp, 'Figures/SegmentationNoiseComp.png')
% As discussed in the document submitted, the algorithms weren't very
% effective dealing with images with noise. Next step was trying to see if
% changing threshold could help, it didn't

ImageGaussian = imnoise(Fruits, 'gaussian'); 
ImageSandP = imnoise(Fruits, 'salt & pepper');

SobelGaussianEdge2 = edge(ImageGaussian, "sobel", 0.1);
SobelSandPEdge2 = edge(ImageSandP, "sobel", 0.1);
PrewittGaussianEdge2 = edge(ImageGaussian, "prewitt", 0.1);
PrewittSandPEdge2 = edge(ImageSandP, "prewitt", 0.1);

SegmentationThresholdNoiseComp = figure;
subplot(3,2,1), imshow(BWsobelOptimal), title('Sobel operator for segmentation')
subplot(3,2,3), imshow(SobelGaussianEdge2), title('Sobel edges on Gaussian noise')
subplot(3,2,5), imshow(SobelSandPEdge2), title('Sobel edges on S and P noise')
subplot(3,2,2), imshow(BWprewittOptimal), title('Prewitt operator for segmentation')
subplot(3,2,4), imshow(PrewittGaussianEdge2), title('Prewitt edges on Gaussian noise')
subplot(3,2,6), imshow(PrewittSandPEdge2), title('Prewitt edges on S and P noise')
saveas(SegmentationThresholdNoiseComp, 'Figures/SegmentationNoiseComp.png')

%Next idea to fix this was by using filters to get noise out

% Median filter
MedFiltGaussianim = medfilt2(ImageGaussian, [5 5]); % [5 5] is superior to default [3 3]
MedFiltSandPim = medfilt2(ImageSandP, [5 5]);

SobelMedFiltGaussianEdge = edge(MedFiltGaussianim, "sobel", 0.02);
PrewittMedFiltGaussianEdge = edge(MedFiltGaussianim, "prewitt", 0.02);
CannyMedFiltGaussianEdge = edge(MedFiltGaussianim, "canny", [0.002 0.01]);
SobelMedFiltSandPEdge = edge(MedFiltSandPim, "sobel", 0.008);
PrewittMedFiltSandPEdge = edge(MedFiltSandPim, "prewitt", 0.008);
CannyMedFiltSandPEdge = edge(MedFiltSandPim, "canny", [0.002 0.01]);

MedianFilterComp = figure;
subplot(3,2,1), imshow(SobelMedFiltGaussianEdge), title('Sobel operator on Median filtered Gaussian noise')
subplot(3,2,3), imshow(PrewittMedFiltGaussianEdge), title('Prewitt operator on Median filtered Gaussian noise')
subplot(3,2,5), imshow(CannyMedFiltGaussianEdge), title('Canny operator on Median filtered Gaussian noise')
subplot(3,2,2), imshow(SobelMedFiltSandPEdge), title('Sobel operator on Median filtered S and P noise')
subplot(3,2,4), imshow(PrewittMedFiltSandPEdge), title('Prewitt operator on Median filtered S and P noise')
subplot(3,2,6), imshow(CannyMedFiltSandPEdge), title('Canny operator on Median filtered S and P noise')
saveas(MedianFilterComp, 'Figures/MedianFilterComp.png')