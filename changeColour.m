function newIm = changeColour(im)
% newIm is impage im flipped from left to right

[nr,nc,np]= size(im);    % dimensions of im
newIm= zeros(nr,nc,np);  % initialize newIm with zeros
newIm= uint8(newIm);     % Matlab uses unsigned 8-bit int for color values


for r= 1:nr
    for c= 1:nc
        if ( im(r,c,1)>180 && im(r,c,2)>180 && im(r,c,3)>180 ) % if RGB values are high enough intensity they are white
            % change white feather of the duck to yellow
            newIm(r,c,1)= 225;
            newIm(r,c,2)= 225;
            newIm(r,c,3)= 0;
        else % keep the rest of th picture the same
            for p= 1:np
                newIm(r,c,p)= im(r,nc-c+1,p);
            end
        end
    end
end