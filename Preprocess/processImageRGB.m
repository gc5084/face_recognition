function [processedImg] = processImageRGB(img, bbox, imageSize)
%This function will resize and make grayscale image of providing image 


% if input image is not grayscale, convert to grayscale
if size(img,3)==3
    processedImg = imresize(imcrop(img, bbox), imageSize);
else
    processedImg = imresize(imcrop(img, bbox), imageSize); 
    processedImg = repmat(processedImg, 1, 1, 3);
end
% imshow(processedImg);
end

