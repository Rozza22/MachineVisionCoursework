close all; 
clear all;

%% Reading image
im = imread('Treasure_medium.jpg'); % change name to process other images
imshow(im);

%% Binarisation
bin_threshold = 0.1; % parameter to vary
bin_im = im2bw(im, bin_threshold);
imshow(bin_im);


%% Extracting connected components
con_com = bwlabel(bin_im);
imshow(label2rgb(con_com));


%% Computing objects properties
props = regionprops(con_com);

%% Drawing bounding boxes
n_objects = numel(props);
imshow(im);
hold on;
for object_id = 1 : n_objects
    rectangle('Position', props(object_id).BoundingBox, 'EdgeColor', 'b');
end
hold off;

%% Arrow/non-arrow determination
% You should develop a function arrow_finder, which returns the IDs of the arror objects. 
% IDs are from the connected component analysis order. You may use any parameters for your function. 

arrow_ind = arrow_finder(props); % returns indicies of objects which are arrows

% Find the differences between consecutive elements
differences = diff(arrow_ind);

% Find the index of the first difference larger than 1
missing_indices = find(differences > 1);

% Initialize an array to store missing numbers
missing_numbers = [];

% Iterate through the missing indices and determine the missing numbers,
% compares arrow_ind to props_ind so we know which lines to draw to yellows
for i = 1:length(missing_indices)
    gap_start = arrow_ind(missing_indices(i));
    gap_end = arrow_ind(missing_indices(i) + 1);
    missing_numbers = [missing_numbers, (gap_start + 1):(gap_end - 1)];
end
% arrows = props(arrow_ind);
% imshow(im);
% hold on;
% for i = 1 : numel(arrow_ind)
%     rectangle('Position', arrows(i).BoundingBox, 'EdgeColor', 'b');
% end
% hold off;

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
[nr,nc,np]= size(im);
newIm2= zeros(nr,nc,np);
newIm2= uint8(newIm2);
for r= 1:nr
    for c= 1:nc
        if im(r,c,1)>25 && im(r,c,2)>25 && im(r,c,3)>25 % if white, turn black
            newIm2(r,c,1)= 0;
            newIm2(r,c,2)= 0;
            newIm2(r,c,3)= 0;
        elseif im(r,c,1)<35 && im(r,c,2)<150 && im(r,c,3)>35  % if blue, turn black
            newIm2(r,c,1)= 0;
            newIm2(r,c,2)= 0;
            newIm2(r,c,3)= 0;
        elseif im(r,c,1)>25 && im(r,c,2)<100 && im(r,c,3)<100  % if red, turn black
            newIm2(r,c,1)= 0;
            newIm2(r,c,2)= 0;
            newIm2(r,c,3)= 0;
        elseif im(r,c,1)<100 && im(r,c,2)>25 && im(r,c,3)<100  % if green, turn black
            newIm2(r,c,1)= 0;
            newIm2(r,c,2)= 0;
            newIm2(r,c,3)= 0;
%         elseif im(r,c,1)>250 && im(r,c,2)>140 && im(r,c,2)<150 && im(r,c,3)<5  % if orange, turn black
%             newIm2(r,c,1)= 0;
%             newIm2(r,c,2)= 0;
%             newIm2(r,c,3)= 0;
        else % the rest of the picture; no change
            for p= 1:np
                newIm2(r,c,p)= im(r,c,p);
            end
        end
    end
end
% figure
imshow(newIm2)

% Make yellow spots black and white for next step ----------------------
bin_threshold2 = 0.2; % parameter to vary 0.2 is better than 0.1 and better than 0.3
bin_im2 = im2bw(newIm2, bin_threshold2);
% figure
% imshow(bin_im2);

% gives each object detected a number ----------------------
con_comYellow = bwlabel(bin_im2); 

% Computing object properties ---------------------------
yellows = regionprops(con_comYellow); 

% separating true yellow spots
% Needed becasue the star is very similar to the yellow dots within arrows

for i = 1 : numel(yellows)
    if yellows(i).Area > 30 && yellows(i).Area < 100
        yellows(i) = yellows(i);
%         yellow_ind(i) = i;
    end
end

yellowCentroids = [yellows(start_arrow_id).Centroid; zeros(numel(yellows)-1,2)];
propsCentroids = [props(start_arrow_id).Centroid; zeros(numel(props)-1,2)];

% yellowCentroids = zeros(numel(yellows),2);
% propsCentroids = zeros(numel(yellows),2);
% Match yellows up with props
% Extract the variable used for ordering from the first struct
for i = 2:numel(yellows) % may have to put blank rows in after this step
    if yellows(i).Centroid == 0
        yellowCentroids(i,:) = [0,0];
    else
        yellowCentroids(i,1) = (yellows(i).Centroid(1,1));
        yellowCentroids(i,2) = (yellows(i).Centroid(1,2));
    end
end

for i = 2:numel(props) % may have to put blank rows in after this step
    propsCentroids(i,1) = (props(i).Centroid(1,1));
    propsCentroids(i,2) = (props(i).Centroid(1,2));
end

% adding blank rows to yellows struct
% Number of blank rows to add
numBlankRows = numel(missing_numbers);
% Create a blank struct with empty fields
% Position to insert the blank rows
positions_to_insert = missing_indices; % could be "missing_indices"
positions_to_insert = sort(positions_to_insert, 'descend');

% Initialize a new array to store the result
newArray = zeros(size(yellowCentroids, 1) + numBlankRows, size(yellowCentroids, 2));

% Initialize insert index for the new array
insertIndex = 1;

% Loop through the original array and insert blank rows
for i = 1:size(yellowCentroids, 1)
    % Insert the current row from the original array into the new array
    newArray(insertIndex, :) = yellowCentroids(i, :);
    
    % Check if a blank row needs to be inserted after this row
    if any(positions_to_insert == i)
        % Insert a blank row
        insertIndex = insertIndex + 1;
    end
    
    % Move to the next insert index in the new array
    insertIndex = insertIndex + 1;
end

yellowCentroids = newArray;

% Make them match up order wise so the yellows are matched with the arrows
% Calculate similarity
distances = pdist2(propsCentroids, yellowCentroids, 'euclidean');

% Sort rows based on similarity
[~, indices] = min(distances, [], 2);

% Reorder matrix based on sorted indices
yellowCentroids = yellowCentroids(indices(:,1), :);

% replace duplicates with 0s
yellowCentroids(missing_numbers, :) = 0;
propsCentroids(missing_numbers, :) = 0;
% Create a logical index to exclude the specified row
% yellow_ind1 = yellow_ind;
% validRows = yellow_ind1(yellow_ind1 ~= 0);

% Remove the specified row from each field
% yellows = yellows(arrow_ind); % This gives us a new struct without the old empty row
% figure


%% add variable to props
% props.yellowCentroid = 'coordinate';
for i = 1:numel(props)
    if yellowCentroids(i,1) ~= 0
        props(i).Centroid = [propsCentroids(i,1), propsCentroids(i,2)];
        props(i).yellowCentroid = [yellowCentroids(i,1), yellowCentroids(i,2)];
    else
        props(i).yellowCentroid = [];
        props(i).Centroid = [];
    end
end

%% show if yellows and arrows are matched
imshow(im);
bboxes = zeros(n_arrows,4);
hold on;
for i = 1 : numel(yellows)
    if yellowCentroids(i) ~= 0
        rectangle('Position', props(i).BoundingBox, 'EdgeColor', 'b'); % error once it gets to blank element in struct
        % bboxes(arrow_id,:) = Arrow(arrow_id).BoundingBox;
        text(props(i).yellowCentroid(1), props(i).yellowCentroid(2), num2str(i), 'Color', 'r')
        text(props(i).BoundingBox(1), props(i).BoundingBox(2), num2str(i), 'Color', 'g')
    end
end
hold off;
%% Hunting
cur_object = start_arrow_id; % start from the red arrow
path = cur_object;

figure
imshow(im)
hold on
i = 1;
% while the current object is an arrow, continue to search
while ismember(cur_object, arrow_ind) 
    % You should develop a function next_object_finder, which returns
    % the ID of the nearest object, which is pointed at by the current
    % arrow. You may use any other parameters for your function.
    
    x1 = round(props(cur_object).Centroid(1,1));
    x2 = round(props(cur_object).yellowCentroid(1,1)); % currently one less yellow point than shapes and need a way of making this right
    y1 = round(props(cur_object).Centroid(1,2));
    y2 = round(props(cur_object).yellowCentroid(1,2));

%     x1 = round(propsCentroids(cur_object,1));
%     x2 = round(yellowCentroids(cur_object,1)); % currently one less yellow point than shapes and need a way of making this right
%     y1 = round(propsCentroids(cur_object,2));
%     y2 = round(yellowCentroids(cur_object,2));

    plot([x1,x2], [y1,y2],'g')

    point1 = [x1,y1];
    point2 = [x2,y2];
    
    % Iteratively extend the line until the condition is satisfied

    % Calculate the next point on the line (you may need to define the line equation)
    next_point = point2 + (point2 - point1) / norm(point2 - point1);
   
    % Check if the pixel at the next point is the same object ID or 0
    x = round(next_point(1));
    y = round(next_point(2));
    pixel_value = con_com(y, x);
    

    % Plot the next line segment
%     if pixel_value == 0 || pixel_value == cur_object
    while pixel_value == 0 || pixel_value == cur_object

        next_point = point2 + (point2 - point1) / norm(point2 - point1);
        % Check if the pixel at the next point is the same object ID or 0
        x = round(next_point(1));
        y = round(next_point(2));
        pixel_value = con_com(y, x);

        plot([point2(1), next_point(1)], [point2(2), next_point(2)], 'Color', 'green');
        
        % Update points for the next iteration
        point1 = point2;
        point2 = next_point; % point2 is final point up to here
    end
    
    text(props(cur_object).BoundingBox(1), props(cur_object).BoundingBox(2), num2str(i), 'Color', 'g')
    rectangle('Position', props(cur_object).BoundingBox, 'EdgeColor', 'b'); % error once it gets to blank element in struct

    % Calculate the next point on the line (you may need to define the line equation)
    next_point = point2 + (point2 - point1) / norm(point2 - point1);
   
    % Check if the pixel at the next point is the same object ID or 0
    x = round(next_point(1));
    y = round(next_point(2));
    pixel_value = con_com(y, x);
    i = i + 1;

    cur_object = pixel_value;

    % cur_object = next_object_finder(props, yellows, con_com, cur_object);
    path(end + 1) = cur_object;
    

end
% hold off

%% visualisation of the path
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
