function [features, setLabels] = helperExtractHOGFeatures(imds, hogFeatureSize, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize, 'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds, j);
    
    if size(img,3) == 3
        % Do grayscale for RGB
        img = rgb2gray(img);
    end
    
    % Apply pre-processing steps
%     imshow(img);
    img = imbinarize(img);
%     imshow(img);
    [features(j, :), visualization] = extractHOGFeatures(img,...
        'CellSize',cellSize);
%     plot(visualization);
end
