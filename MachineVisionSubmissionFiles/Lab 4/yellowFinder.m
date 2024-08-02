function  props = yellowFinder(props, arrow_ind, im)
    [nr,nc,np]= size(im);
    newIm2= zeros(nr,nc,np);
    newIm2= uint8(newIm2); 
    % had to do image processing to eliminate other yellowish colours to
    % leave us only with the yellow spots
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
            else % the rest of the picture; no change
                for p= 1:np
                    newIm2(r,c,p)= im(r,c,p);
                end
            end
        end
    end
    % figure
%     imshow(newIm2)
    
    % Make yellow spots black and white for next step ----------------------
    bin_threshold2 = 0.2; % parameter to vary 0.2 is better than 0.1 and better than 0.3
    bin_im2 = im2bw(newIm2, bin_threshold2);
    % figure
%     imshow(bin_im2);
    
    % gives each object detected a number ----------------------
    con_comYellow = bwlabel(bin_im2); 
    
    % Computing object properties ---------------------------
    yellows = regionprops(con_comYellow); 
    
    % separating true yellow spots
    % Needed becasue the star is very similar to the yellow dots within arrows
    
    for i = 1 : numel(yellows)
        if yellows(i).Area > 30 && yellows(i).Area < 100
            yellows(i) = yellows(i);
            yellow_ind(i) = i;
        end
    end
    
    % Create a logical index to exclude the specified row
    validRows = yellow_ind(yellow_ind ~= 0);
    
    % Remove the specified row from each field
    yellows = yellows(validRows); % This gives us a new struct without the old empty row
    
    yellowCentroids = zeros(numel(yellows),2);
    propsCentroids = zeros(numel(yellows),2);
    % Match yellows up with props
    % Extract the variable used for ordering from the first struct
    for i = 1:numel(yellows) % may have to put blank rows in after this step
        if yellows(i).Centroid == 0
            yellowCentroids(i,:) = [0,0];
        else
            yellowCentroids(i,1) = (yellows(i).Centroid(1,1));
            yellowCentroids(i,2) = (yellows(i).Centroid(1,2));
        end
    end
    
    for i = 1:numel(props) % may have to put blank rows in after this step
        propsCentroids(i,1) = (props(i).Centroid(1,1));
        propsCentroids(i,2) = (props(i).Centroid(1,2));
    end
    
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
    
    % Reorder the remaining rows to maintain the original order
    propsCentroids = [propsCentroids(1, :); propsCentroids(setdiff(1:size(propsCentroids, 1), 1), :)];
    
    % Display the updated array
    disp(propsCentroids);
    
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
    
    
    % add variable to props
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
end