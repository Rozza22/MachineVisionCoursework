% Script for extra report tasks
close all; 
clear

%% Reading image
im1 = imread('Treasure_easy.jpg'); % change name to process other images
im2 = imread('Treasure_medium.jpg'); % change name to process other images
im3 = imread('Treasure_hard.jpg'); % change name to process other images

threeTreasures = figure;
subplot(1,3,1), imshow(im1), title('Easy');
subplot(1,3,2), imshow(im2), title('Medium');
subplot(1,3,3), imshow(im3), title('Hard');
saveas(threeTreasures,'Figures/threeTreasures.png')

%% Binarisation
bin_threshold1 = 0; % parameter to vary
bin_threshold2 = 0.1; % parameter to vary
bin_threshold3 = 0.2; % parameter to vary

bin1_im1 = im2bw(im1, bin_threshold1);
bin2_im1 = im2bw(im1, bin_threshold2);
bin3_im1 = im2bw(im1, bin_threshold3);

varyingThresholds = figure;
subplot(1,3,1), imshow(bin1_im1), title('Threshold = 0');
subplot(1,3,2), imshow(bin2_im1), title('Threshold = 0.1');
subplot(1,3,3), imshow(bin3_im1), title('Threshold = 0.2');
saveas(varyingThresholds,'Figures/varyingThresholds.png')