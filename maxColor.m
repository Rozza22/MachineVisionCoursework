function [r, g, b] = maxColor(im)

imageRed = im(:,:,1);
imageGreen = im(:,:,2);
imageBlue = im(:,:,3);

r = max(imageRed(:));
g = max(imageGreen(:));
b = max(imageBlue(:));