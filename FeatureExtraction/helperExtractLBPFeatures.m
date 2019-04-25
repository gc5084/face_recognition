function [features, labels] = helperExtractLBPFeatures(imds)
% Input image is grayscale 

labels = imds.Labels;
numImages = numel(imds.Files);
features  = [];

% Process each image and extract features
for j = 1:numImages
      grayImage = readimage(imds, j); 
      features(j, :) = extractLBPFeatures(grayImage, 'Normalization', 'L2');

end