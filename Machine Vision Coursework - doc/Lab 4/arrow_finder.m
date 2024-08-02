function arrow_ind = arrow_finder(props) % takes struct of all shapes with characteristics as input
    n_objects = numel(props);
    arrow_ind = zeros(size(n_objects));
    for object_id = 1 : n_objects % For the number of objects identified, this is carried out
        if props(object_id).Area > 1430 && props(object_id).Area < 1600 % Arrows fall within this area, other objects don't
            arrow_ind(object_id) = object_id;
        end
    end
    arrow_ind = nonzeros(arrow_ind); % outputs the indicies of the arrows in an order which is not useful to us
end


