function [identityId] = my_face_recognition_function(image, faceRecoguntionModel)


bbox = MyFaceDetectionFunction(faceRecoguntionModel.faceDetector, image);

% !!! Tunable parameter or required settings
% cellSize, image size

% Get number of detected face
if ~isempty(bbox) == 1
    
%     [imageSize, hogCellSize, pcaCoeff, pcaNumComponents] = getGlobalVar();
    % Face is deteced then do following:
    % ----- preprocess images
    % ----- feature extraction (HOG, PCA)
    % ----- Predict by the model
    
    % FOR CNN NN
    % identityId = classify(faceRecoguntionModel,image);
    %YTest = imdsTest.Labels;
    
    
    % FOR SVM - CNN
    %     rgbImage = imresize(imcrop(image, bbox), faceRecoguntionModel.imageSize);
    %     if size(rgbImage,3) ~= 3
    %         rgbImage=rgbImage(:,:,[1 1 1]);
    %     end
    
    %     modelFile = 'cnnModelLab.mat';
    %     load(modelFile, 'cnnModelLab');
    %     net = vgg16;
    %     featuresTest = activations(net,rgbImage,'fc7','OutputAs','rows');
    %     impostorCheck = predict(cnnModelLab, featuresTest);
    %     if impostorCheck ~= num2str(-1)
    %     % 1. Preprocess images
    
    processedImage = processImageGrayscale(image, bbox, ...
        faceRecoguntionModel.imageSize);
    
    % 2. Feature extraction
    dataProjected = extractFeature(processedImage, ...
        faceRecoguntionModel.cellSize, faceRecoguntionModel.pcaCoeff);
    
    % 3. Predict by the model
    % !!! Change to Jess function here.
    [label, ~] = predict(faceRecoguntionModel.classifier, dataProjected);
    
    identityId = grp2idx(label);
    
    labelLen = length(label);
    convertedPredictedLabel = [];
    for c = 1:labelLen
        convertedPredictedLabel = [convertedPredictedLabel,char(label(c))];
    end
    identityId = str2double(convertedPredictedLabel);
    % ============ Thresholding Start =================
%     imgPath = "/Volumes/Work/UPF/Class_FACIAL/FaceAndGesture-Lab4/AGC2019_Challenge3_Materials/Models/DS/AugmentedGrayscale128/" + convertedLabel + "/";
%     
%     imdsPredictedLabel = imageDatastore(imgPath, ...
%         'IncludeSubfolders',true,'FileExtensions',{'.jpg'});
    
    
%     imgTemp = imread(imdsPredictedLabel.Files{1});
%     [hog, ~] = extractHOGFeatures(imgTemp,'CellSize',hogCellSize);
%     hogFeatureSize = length(hog);
%     [X, ~] = helperExtractHOGFeatures(imdsPredictedLabel, 2304, hogCellSize);
%     
%     subSpaceOfPredictedLabel = projectData(X, pcaCoeff);
%     S = cov(subSpaceOfPredictedLabel);
%     mu = mean(subSpaceOfPredictedLabel,1);
    
%     d = sqrt((dataProjected-mu)*inv(S)*(dataProjected-mu)');
%     eu_distances = 'euclidean_distances.mat';
%     load(eu_distances);
%     % get max distance
%     maxDistance = group_distances(str2num(convertedPredictedLabel), 3);
%     observeDistance = pdist([group_distances(str2num(convertedPredictedLabel),4:503); dataProjected],'euclidean');
%     fprintf('predicted: %s max: %.2f\t distance: %.2f\n', ...
%         convertedPredictedLabel, maxDistance, observeDistance);
%     if(observeDistance > (maxDistance + 3))
%         identityId = -1;
%     else
%         identityId = str2double(convertedPredictedLabel);
%     end

    % identityId = TestImpostor(faceRecoguntionModel, image);
    % ============ Thresholding End =================
    
    
else
    identityId = -1;
end

end

