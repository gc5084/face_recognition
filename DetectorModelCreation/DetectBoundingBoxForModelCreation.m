function [bbox] = DetectBoundingBoxForModelCreation(detector, img)
% MyFaceDetectionFunction detects image for human face by using providing
% detector model then calculate x2, y2 due to the result from detecting
% return width and height but program result required x and y postion.

% Detect face boundary box.

% Test converting image to grayscale but accuracy doesn't increase.
% if size(img,3) == 3
%     img = rgb2gray(img);
% end
bbox = detector(img);

% Get mumber of box in image.
s_b = size(bbox);
% IFaces = insertObjectAnnotation(img,'rectangle',bbox,'area');   
% figure;
% imshow(IFaces);
% Loop to modify value of width and height to x2, y2.
for line = 1: s_b(1)
    bbox(line,3) = bbox(line,1)+ bbox(line,3);
    bbox(line,4) = bbox(line,2)+ bbox(line,4);
end
%  IFaces = insertObjectAnnotation(img,'rectangle',bbox,'Face');   
%  figure;
%  imshow(IFaces);
end

