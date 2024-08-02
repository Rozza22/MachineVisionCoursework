function cur_object = next_object_finder(props, con_com, cur_object)

    x1 = round(props(cur_object).Centroid(1,1));
    x2 = round(props(cur_object).yellowCentroid(1,1)); % currently one less yellow point than shapes and need a way of making this right
    y1 = round(props(cur_object).Centroid(1,2));
    y2 = round(props(cur_object).yellowCentroid(1,2));

%     plot([x1,x2], [y1,y2],'g')

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

%        plot([point2(1), next_point(1)], [point2(2), next_point(2)], 'Color', 'green');
        
        % Update points for the next iteration
        point1 = point2;
        point2 = next_point; % point2 is final point up to here
    end
    
%     text(props(cur_object).BoundingBox(1), props(cur_object).BoundingBox(2), num2str(i), 'Color', 'g')
%     rectangle('Position', props(cur_object).BoundingBox, 'EdgeColor', 'b'); % error once it gets to blank element in struct

    % Calculate the next point on the line (you may need to define the line equation)
    next_point = point2 + (point2 - point1) / norm(point2 - point1);
   
    % Check if the pixel at the next point is the same object ID or 0
    x = round(next_point(1));
    y = round(next_point(2));
    pixel_value = con_com(y, x);

    cur_object = pixel_value;
end
