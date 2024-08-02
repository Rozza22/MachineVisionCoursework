% script for lab 1

%% part 1
clear
close all;

im1 = imread('Images\Beautiful_Green_Picture.jpg');
figure
imshow(im1)
imwrite(im1,'Figures/origninalImage.png')

% Extract RGBs frame of data and display
figure
imageB = im1(:,:,3);
subplot(1,3,3)
imshow(imageB)
title('Image Data - Blue')

imageG = im1(:,:,2);
subplot(1,3,2)
imshow(imageG)
title('Image Data - Green')

imageR = im1(:,:,1);
subplot(1,3,1)
imshow(imageR)
title('Image Data - Red')

% Histograms generation

imageWRGBhist = figure; %step needed to save figure
% subplot(3,2,[1 3 5])
% imshow(im1)

[countsR, binLocR] = imhist(imageR);
subplot(3,1,1)
%histogram(histR, nbins)
imhist(imageR)
title('Red histogram')

[countsG, binLocG] = imhist(imageG);
subplot(3,1,2)
%histogram(histG, nbins)
imhist(imageG)
title('Green histogram')

[countsB, binLocB] = imhist(imageB);
subplot(3,1,3)
%histogram(histB, nbins)
imhist(imageB)
title('Blue histogram')
saveas(imageWRGBhist,'Figures/imageWRGBhist.png')

% Finding median to binarize the colours
BinaryR1 = imbinarize(imageR,"global"); % Uses "Otsu's method"
BinaryR2 = im2bw(imageR, 128/255); % Vary threshold and comment on results
figure
subplot(2,1,1) % Make more and compare
imshow(BinaryR1)
subplot(2,1,2)
imshow(BinaryR2)

% Calculate and visualise the histogram of a HSV image

% Display the original image.
figure
subplot(2, 4, 1);
imshow(im1, [ ]);
title('Original RGB image');
% Convert to HSV color space
hsvimage = rgb2hsv(im1);
% Extract out the individual channels.
hueImage = hsvimage(:,:,1);
satImage = hsvimage(:,:,2);
valueImage = hsvimage(:,:,3);
% Display the individual channels.
subplot(2, 4, 2);
imshow(hueImage, [ ]);
title('Hue Image');
subplot(2, 4, 3);
imshow(satImage, [ ]);
title('Saturation Image');
subplot(2, 4, 4);
imshow(valueImage, [ ]);
title('Value Image');
% Take histograms
[hCount, hValues] = imhist(hueImage(:), 18);
[sCount, sValues] = imhist(satImage(:), 3);
[vCount, vValues] = imhist(valueImage(:), 3);
% Plot histograms.
subplot(2, 4, 6);
bar(hValues, hCount);
title('Hue Histogram');
subplot(2, 4, 7);
bar(sValues, sCount);
title('Saturation Histogram');
subplot(2, 4, 8);
bar(vValues, vCount);
title('Value Histogram');
% Alert user that we're done.
message = sprintf('Done processing this image.\n Maximize and check out the figure window.');
msgbox(message);

% Final section of part 1
oneColour = imread('One_colour.jpg');
twoColour = imread('Two_colour.jpg');

oneCGrey = rgb2gray(oneColour);
twoCGrey = rgb2gray(twoColour);

figure
subplot(2,1,1), imhist(oneCGrey)
subplot(2,1,2), imhist(twoCGrey)

oneCR = oneColour(:,:,1);
oneCG = oneColour(:,:,2);
oneCB = oneColour(:,:,3);

twoCR = twoColour(:,:,1);
twoCG = twoColour(:,:,2);
twoCB = twoColour(:,:,3);

figure
subplot(2,3,1), imhist(oneCR), title('One Colour - Red')
subplot(2,3,2), imhist(oneCG), title('One Colour - Green')
subplot(2,3,3), imhist(oneCB), title('One Colour - Blue')
subplot(2,3,4), imhist(twoCR), title('Two Colour - Red')
subplot(2,3,5), imhist(twoCG), title('Two Colour - Green')
subplot(2,3,6), imhist(twoCB), title('Two Colour - Blue')


%% part 2 - Edge Detection and Segmentation of static objects
clear

% 1 -----------------------------------------------------------------------
Fruits = imread('Images\Fruits.jpg');
Fruits = rgb2gray(Fruits);
imwrite(Fruits,'Figures\imageFruits.png')

% 2 and 3 -----------------------------------------------------------------
equalisedFruits = histeq(Fruits); % histogram equalisation

figure
subplot(2,2,1), imshow(Fruits), title('Original Image')
subplot(2,2,3), imshow(equalisedFruits), title('Equalised Image')

subplot(2,2,2), imhist(Fruits), title('Original Histogram')
subplot(2,2,4), imhist(equalisedFruits), title('Equalised Histogram')

OriginalGrFruitsScr = piqe(Fruits); % smaller score is better in terms of "perceptual quality"
equalisedFruitsScr = piqe(equalisedFruits);

% 4 -----------------------------------------------------------------------
attempts = 20;
gamma = linspace(0,2,attempts); % If higher than 1, the image becomes darker
gammaCorrecFruitsScr = zeros(attempts,1);

for i = 1:attempts
    gammaCorrecFruits = imadjust(Fruits,[0 1],[0 1],gamma(i));
    gammaCorrecFruitsScr(i) = piqe(gammaCorrecFruits);
end

[OptimalScr, pos] = min(gammaCorrecFruitsScr,[],1);
OptimalGamma = gamma(:,13);

gammaCorrecFruits = imadjust(Fruits,[0 1],[0 1],OptimalGamma); % both [0 1] inputs are defaults and mean its for greyscale image
figure
subplot(2,2,1), imshow(Fruits), title('Original Image')
subplot(2,2,3), imshow(gammaCorrecFruits), title('Gamma corrected Image')

subplot(2,2,2), imhist(Fruits), title('Original Histogram')
subplot(2,2,4), imhist(gammaCorrecFruits), title('Gamma corrected Histogram')

% 5 -----------------------------------------------------------------------
ImageGaussian = imnoise(Fruits, 'gaussian'); %adding noise to images
ImageSandP = imnoise(Fruits, 'salt & pepper');

figure
subplot(2,2,1), imshow(ImageGaussian), title('Image with Gaussian Synthesisation'), zoom(4)
subplot(2,2,2), imshow(ImageSandP), title('Image with Salt and Pepper Synthesisation'), zoom(4)

% 6 -----------------------------------------------------------------------
sigma = 0.8; % 0.5 is default
GaussFiltGaussIm = imgaussfilt(ImageGaussian, sigma);
% 7 -----------------------------------------------------------------------
GaussFiltSandPIM = imgaussfilt(ImageSandP, sigma);

subplot(2,2,3), imshow(GaussFiltGaussIm), title('Gaussian Filter applied on image with gaussian noise'), zoom(4)
subplot(2,2,4), imshow(GaussFiltSandPIM), title('Gaussian Filter applied on image with S and P noise'), zoom(4)

% 8 -----------------------------------------------------------------------
MedFiltSandPim1 = medfilt2(ImageSandP, [3 3]); % This is default neighbourhood size
MedFiltSandPim2 = medfilt2(ImageSandP, [1 1]);
MedFiltSandPim3 = medfilt2(ImageSandP, [5 5]);

figure
subplot(2,2,1), imshow(ImageSandP), title('Image with S and P noise')
subplot(2,2,2), imshow(MedFiltSandPim1), title('Image with default median filter')
subplot(2,2,3), imshow(MedFiltSandPim2), title('Image with smaller neighbourhood size median filter')
subplot(2,2,4), imshow(MedFiltSandPim3), title('Image with larger neighbourhood size median filter')

% 9 -----------------------------------------------------------------------
BWsobel = edge(equalisedFruits, "sobel", 0.03, "nothinning"); % This uses the sobel edge detection method

% 10 ----------------------------------------------------------------------
BWprewitt = edge(equalisedFruits, "prewitt", 0.03, "nothinning");

% 11 ----------------------------------------------------------------------
BWcanny = edge(equalisedFruits, "canny", [0.025 0.1], "nothinning");

figure
subplot(2,2,1), imshow(Fruits), title('Orignal image')
subplot(2,2,2), imshow(BWsobel), title('Sobel operator for segmentation')
subplot(2,2,3), imshow(BWprewitt), title('Prewitt operator for segmentation')
subplot(2,2,4), imshow(BWcanny), title('Canny operator for segmentation')

