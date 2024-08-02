close all; 
clear all;

%% Reading image
im = imread('Treasure_easy.jpg'); % change  file name to process other images - this works for all
% imshow(im);

%% Binarisation
bin_threshold = 0.1; % parameter to vary
bin_im = im2bw(im, bin_threshold); % Separates shapes from background
% imshow(bin_im);


%% Extracting connected components
con_com = bwlabel(bin_im); % Puts labels on the shapes
% imshow(label2rgb(con_com));

%% Computing objects properties
props = regionprops(con_com); % Function computes some different shape properties, including area

%% Drawing bounding boxes
n_objects = numel(props);
% imshow(im);
hold on;
for object_id = 1 : n_objects
    rectangle('Position', props(object_id).BoundingBox, 'EdgeColor', 'b');
end
hold off;

%% Arrow/non-arrow determination
% You should develop a function arrow_finder, which returns the IDs of the arror objects. 
% IDs are from the connected component analysis order. You may use any parameters for your function. 

arrow_ind = arrow_finder(props); % returns indicies of objects which are arrows

arrows = props(arrow_ind);
% arrowFinder = figure; % commented out, but this was to check everything was going well
% imshow(im);
% hold on;
% for i = 1 : numel(arrow_ind)
%     rectangle('Position', arrows(i).BoundingBox, 'EdgeColor', 'g');
% end
% hold off;
% saveas(arrowFinder,'Figures/arrowFinder.png')

%% Finding red arrow
n_arrows = numel(arrow_ind);
start_arrow_id = 0;
% check each arrow until find the red one
for arrow_num = 1 : n_arrows
    object_id = arrow_ind(arrow_num);    % determine the arrow id
    
    % extract colour of the centroid point of the current arrow
    centroid_colour = im(round(props(object_id).Centroid(2)), round(props(object_id).Centroid(1)), :); 
    if centroid_colour(:, :, 1) > 240 && centroid_colour(:, :, 2) < 10 && centroid_colour(:, :, 3) < 10
	% the centroid point is red, memorise its id and break the loop
        start_arrow_id = object_id; % object ID of arrow
        break;
    end
end

%% Find yellow spots to allow us to tell direction of arrow

% This function adds a new field (which has the centroid of the yellow point) to the props struct
props = yellowFinder(props, arrow_ind, im);

%% show if yellows and arrows are matched - another check
% imshow(im);
% bboxes = zeros(n_arrows,4);
% hold on;
% for i = 1 : numel(props)
%     if props(i).yellowCentroid ~= 0
%         text(props(i).yellowCentroid(1), props(i).yellowCentroid(2), num2str(i), 'Color', 'r')
%         text(props(i).Centroid(1), props(i).Centroid(2), num2str(i), 'Color', 'b')
%     end
% end
% hold off;
%% Hunting
cur_object = start_arrow_id; % start from the red arrow
path = cur_object;

% while the current object is an arrow, continue to search
while ismember(cur_object, arrow_ind) 
    cur_object = next_object_finder(props, con_com, cur_object);
    path(end + 1) = cur_object;
end
% hold off

%% visualisation of the path
HardVisualSolution = figure;
imshow(im);
hold on;
for path_element = 1 : numel(path) - 1
    object_id = path(path_element); % determine the object id
    rectangle('Position', props(object_id).BoundingBox, 'EdgeColor', 'y');
    str = num2str(path_element);
    text(props(object_id).BoundingBox(1), props(object_id).BoundingBox(2), str, 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 14);
end

% visualisation of the treasure
treasure_id = path(end);
rectangle('Position', props(treasure_id).BoundingBox, 'EdgeColor', 'g');
saveas(HardVisualSolution,'Figures/HardVisualSolution.png')
