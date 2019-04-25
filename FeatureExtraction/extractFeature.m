function dataProjected = extractFeature(image, cellSize, coeff)
% extract feature for query image.

    % Apply pre-processing steps
    img = imbinarize(image);
    features = extractHOGFeatures(img,'CellSize', cellSize);
    dataProjected = projectData(features, coeff);
    
end

