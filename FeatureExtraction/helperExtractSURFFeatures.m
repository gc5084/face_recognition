function [features, setLabels] = helperExtractSURFFeatures(imds)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = [];

% Process each image and extract features
for j = 1:numImages
    
    img = readimage(imds, j);

    if size(img,2) == 3
        % Do grayscale for RGB
        img = rgb2gray(img);
    end
    
    % Apply pre-processing steps
    %img = imbinarize(img);
    points = detectSURFFeatures(img);
    imshow(img); hold on; plot(points);
    features(j, :, :) = points.Location;
  
end
